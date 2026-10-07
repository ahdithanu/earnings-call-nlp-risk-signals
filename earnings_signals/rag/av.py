"""Alpha Vantage earnings-call-transcript ingestion: parsing + Q&A isolation.

The EARNINGS_CALL_TRANSCRIPT endpoint returns transcripts ALREADY
segmented into speaker turns with titles:

    {"symbol": "IBM", "quarter": "2024Q1",
     "transcript": [{"speaker": "...", "title": "...", "content": "...",
                     "sentiment": "..."}]}

so no format regexes are needed — this module turns that payload into the
same (speaker, words) executive-Q&A turns the chunker consumes from the
other sources. The rules mirror the Motley Fool parser: the Q&A starts at
the first analyst-titled turn, and inside it analyst and operator turns
are dropped by title/name.

Error shapes the endpoint is known to return instead of data (all HTTP
200): {"Information": "...rate limit..."}, {"Note": "..."}, or an
"Error Message" for a bad symbol — classified here so the fetch script
can budget and retry correctly.

The network, caching and budgeting live in scripts/fetch_av_transcripts.py;
this module is pure and unit-tested.
"""

from dataclasses import dataclass

_ANALYST = "analyst"
_OPERATOR = "operator"


@dataclass
class AVResult:
    status: str  # "ok" | "empty" | "rate_limited" | "error"
    turns: list[dict]  # raw transcript turns when status == "ok"
    detail: str = ""


def parse_av_response(payload: object) -> AVResult:
    """Classify one EARNINGS_CALL_TRANSCRIPT response body."""
    if not isinstance(payload, dict):
        return AVResult("error", [], f"unexpected payload type {type(payload).__name__}")
    for key in ("Information", "Note"):
        if key in payload:
            msg = str(payload[key])
            status = "rate_limited" if "limit" in msg.lower() else "error"
            return AVResult(status, [], msg[:200])
    if "Error Message" in payload:
        return AVResult("error", [], str(payload["Error Message"])[:200])
    turns = payload.get("transcript")
    if not isinstance(turns, list) or not turns:
        return AVResult("empty", [], "no transcript for this symbol/quarter (yet)")
    ok = [t for t in turns if isinstance(t, dict) and t.get("content")]
    if not ok:
        return AVResult("empty", [], "transcript list had no usable turns")
    return AVResult("ok", ok)


def exec_qa_turns_from_av(turns: list[dict]) -> list[tuple[str, str]]:
    """Executive Q&A turns from structured AV turns (same contract as the
    other parsers: [] when no Q&A boundary or no attributable turns)."""
    qa_from = next(
        (i for i, t in enumerate(turns) if _ANALYST in str(t.get("title", "")).lower()),
        None,
    )
    if qa_from is None:
        return []
    out: list[tuple[str, str]] = []
    for t in turns[qa_from:]:
        title = str(t.get("title", "")).lower()
        speaker = str(t.get("speaker", "")).strip()
        if _ANALYST in title or _OPERATOR in title or speaker.lower() == _OPERATOR:
            continue
        words = str(t.get("content", "")).strip()
        if words:
            out.append((speaker.lower(), words))
    return out
