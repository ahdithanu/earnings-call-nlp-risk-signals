"""Speaker-turn chunking of executive Q&A speech.

The retrieval unit is one executive answer in the Q&A — the same unit the
isolation layer (earnings_signals/qa_isolation.py) already attributes. A
turn is a semantically coherent chunk by construction: one speaker, one
question being answered, so no arbitrary fixed-size windows are needed.
Two adjustments keep chunk quality even:

  - turns shorter than ``MIN_TURN_TOKENS`` are dropped ("Yes." / "Thanks,
    next question" carry no retrievable content and pollute rankings);
  - turns longer than ``MAX_CHUNK_TOKENS`` are split at sentence
    boundaries into pieces of roughly ``TARGET_CHUNK_TOKENS``, so one
    monologue can't dominate a single embedding.

Every chunk carries the metadata retrieval filters on: ticker, sector,
quarter, and the (normalized) speaker name.

Two transcript formats are handled, auto-detected per call:

  - Seeking Alpha style ("Name : words", roster header) — the validated
    panel source; parsed by earnings_signals/qa_isolation.py.
  - Motley Fool style ("Name\\n--\\nTitle\\n\\nwords" turn headers) — the
    live Rogersurf source. Titles ride along with every turn, so executive
    attribution is direct: drop turns titled Analyst and the Operator,
    keep the rest, starting at the "Questions & Answers" marker (or the
    first analyst-titled turn when the marker is missing).

Measured on the 1,256 live calls beyond the panel watermark: the colon
parser handles 86%, the Motley Fool parser another 12%; 2% parse as
neither and yield no chunks (consistent with how the feature pipeline
treats attribution failure).
"""

import re
from dataclasses import asdict, dataclass

from earnings_signals.qa_isolation import executive_qa_turns

MIN_TURN_TOKENS = 20
TARGET_CHUNK_TOKENS = 120
MAX_CHUNK_TOKENS = 180

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")

# Turn labels like "Thanks. Tim Cook :" normalize to "thanks tim cook";
# strip the courtesy prefix so speaker metadata stays a name.
_SPEAKER_PREFIX_RE = re.compile(r"^(?:thanks?|thank you|yes|yeah|sure|okay|ok)\s+")

# Motley Fool turn header: speaker name, "--" on its own line, then title.
_MF_HEADER_RE = re.compile(r"\n([^\n]{2,60})\n--\n([^\n]{2,90})\n")
_MF_MIN_HEADERS = 5  # fewer and the call is not in this format
_MF_QA_RE = re.compile(r"questions?\s*(?:&(?:amp;)?|and)\s*answers?", re.IGNORECASE)


def _norm_name(name: str) -> str:
    return " ".join(re.findall(r"[a-zà-ÿ]+", name.lower()))


def _mf_exec_qa_turns(transcript: str) -> list[tuple[str, str]] | None:
    """Executive Q&A turns of a Motley Fool-format call.

    Returns None when the transcript is not in this format (caller should
    try the colon parser), and [] when it is but has no attributable
    executive Q&A turns.
    """
    headers = list(_MF_HEADER_RE.finditer(transcript))
    if len(headers) < _MF_MIN_HEADERS:
        return None

    qa_marker = _MF_QA_RE.search(transcript)
    qa_start: int | None
    if qa_marker is not None:
        qa_start = qa_marker.start()
    else:
        qa_start = next(
            (h.start() for h in headers if "analyst" in h.group(2).lower()),
            None,
        )
    if qa_start is None:
        return []

    turns: list[tuple[str, str]] = []
    for i, h in enumerate(headers):
        if h.start() < qa_start:
            continue
        name, title = h.group(1).strip(), h.group(2).strip().lower()
        if "analyst" in title or "operator" in (title, name.lower()):
            continue
        end = headers[i + 1].start() if i + 1 < len(headers) else len(transcript)
        words = transcript[h.end() : end].strip()
        if words:
            turns.append((_norm_name(name), words))
    return turns


@dataclass
class Chunk:
    chunk_id: str  # "<TICKER>-<YYYY>Q<Q>-<seq>"
    ticker: str
    company: str
    sector: str
    year: int
    quarter: int
    speaker: str  # normalized speaker name ("" if unknown)
    text: str
    source: str = ""  # transcript provenance, e.g. "panel" | "rogersurf"

    @property
    def quarter_label(self) -> str:
        return f"{self.year}Q{self.quarter}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long_turn(text: str) -> list[str]:
    """Split one long turn at sentence boundaries into ~TARGET-token pieces."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for m in _SENTENCE_RE.finditer(text):
        sentence = m.group(0).strip()
        if not sentence:
            continue
        n = len(sentence.split())
        if current and current_tokens + n > TARGET_CHUNK_TOKENS:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunks_from_turns(
    turns: list[tuple[str, str]],
    *,
    ticker: str,
    company: str,
    sector: str,
    year: int,
    quarter: int,
    source: str = "",
) -> list[Chunk]:
    """Retrieval chunks from already-attributed (speaker, words) exec turns.

    The shared back half of every ingestion path: the two text parsers
    below feed it, and so does the Alpha Vantage path (rag/av.py), whose
    API returns pre-segmented turns that need no parsing at all.
    """
    chunks: list[Chunk] = []
    seq = 0
    for speaker, words in turns:
        if len(words.split()) < MIN_TURN_TOKENS:
            continue
        speaker = _SPEAKER_PREFIX_RE.sub("", speaker)
        parts = [words] if len(words.split()) <= MAX_CHUNK_TOKENS else _split_long_turn(words)
        for part in parts:
            if len(part.split()) < MIN_TURN_TOKENS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{ticker}-{year}Q{quarter}-{seq:03d}",
                    ticker=ticker,
                    company=company,
                    sector=sector,
                    year=year,
                    quarter=quarter,
                    speaker=speaker,
                    text=part,
                    source=source,
                )
            )
            seq += 1
    return chunks


def chunk_transcript(
    transcript: str,
    *,
    ticker: str,
    company: str,
    sector: str,
    year: int,
    quarter: int,
    source: str = "",
) -> list[Chunk]:
    """Executive Q&A turns of one call as retrieval chunks (possibly empty).

    Calls where the Q&A boundary or executive attribution fails yield no
    chunks — consistent with the feature pipeline, which treats those
    scopes as missing rather than substituting the full transcript.
    """
    mf_turns = _mf_exec_qa_turns(transcript)
    if mf_turns is not None:
        turns = mf_turns
    else:
        turns, mode = executive_qa_turns(transcript)
        if mode != "exec_turns":
            return []
    return chunks_from_turns(
        turns,
        ticker=ticker,
        company=company,
        sector=sector,
        year=year,
        quarter=quarter,
        source=source,
    )
