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


def chunk_transcript(
    transcript: str,
    *,
    ticker: str,
    company: str,
    sector: str,
    year: int,
    quarter: int,
) -> list[Chunk]:
    """Executive Q&A turns of one call as retrieval chunks (possibly empty).

    Calls where the Q&A boundary or executive attribution fails yield no
    chunks — consistent with the feature pipeline, which treats those
    scopes as missing rather than substituting the full transcript.
    """
    turns, mode = executive_qa_turns(transcript)
    if mode != "exec_turns":
        return []
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
                )
            )
            seq += 1
    return chunks
