"""Executive-only answer isolation within the Q&A section.

``qa_extract`` finds where the Q&A starts; this module goes one step further
and keeps ONLY the text spoken by company executives inside it. The research
hypothesis is about *executive* hedging, and analyst questions are
disproportionately loaded with uncertainty terms ("what risks do you see",
"could margins vary"), so scoring the whole Q&A section mixes the signal
with analyst phrasing. The ``_execqa`` metrics built from this module remove
that contamination at the source.

Speaker structure: transcripts mark turns as ``Name : text`` (analyst labels
often ``Name - Firm:``). Executive identification combines two independent
sources:
  1. the ``Executives:`` roster in the header when present (catches execs
     who only join for Q&A), and
  2. every speaker who talks BEFORE the Q&A boundary — prepared remarks are
     delivered exclusively by company-side speakers — minus the Operator.
Rostered analysts and the Operator are always excluded; a Q&A speaker not
identifiable as an executive is treated as an analyst and excluded.
"""

import re
from dataclasses import dataclass

from src.qa_extract import find_qa_start

# A speaker turn label: 1-4 capitalized words (accents/initials/hyphens
# allowed), optionally followed by "- Firm Name" (analyst labels are often
# "Mike Wood - Macquarie:"), ending in a colon. The captured speaker is the
# name only. Matched with finditer to split text into turns; text before the
# first label has no speaker and is dropped.
_TURN_RE = re.compile(
    r"(?:^|\s)([A-Z][\w.'’\-]*(?:\s+[A-Z(][\w.'’\-)]*){0,3})"
    r"(?:\s*[–—-]\s*[A-Z][\w.'’&()\-]*(?:\s+[A-Z][\w.'’&()\-]*){0,4})?\s*:",
    re.UNICODE,
)

# Roster entries look like "Bill Sullivan - President and CEO"; the name is
# the capitalized run immediately before a dash.
_ROSTER_NAME_RE = re.compile(
    r"([A-Z][\w.'’\-]*(?:\s+[A-Z][\w.'’\-]*){0,3})\s*[-–—]", re.UNICODE
)


def _norm(name: str) -> str:
    """Normalize a speaker name for matching: lowercase, letters only."""
    return " ".join(re.findall(r"[a-zà-ÿ]+", name.lower()))


def _parse_roster(header: str, start_label: str, end_labels: list[str]) -> set[str]:
    """Extract normalized names from an 'Executives:'/'Analysts:' roster block."""
    m = re.search(start_label, header, re.IGNORECASE)
    if not m:
        return set()
    block = header[m.end():]
    end = len(block)
    for lbl in end_labels:
        e = re.search(lbl, block, re.IGNORECASE)
        if e:
            end = min(end, e.start())
    block = block[:end]
    return {_norm(n) for n in _ROSTER_NAME_RE.findall(block)}


def _split_turns(text: str) -> list[tuple[str, str]]:
    """Split ``Name : words`` formatted text into (normalized_speaker, words)."""
    matches = list(_TURN_RE.finditer(text))
    turns = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append((_norm(m.group(1)), text[m.end():end].strip()))
    return turns


@dataclass
class ExecQAResult:
    text: str  # concatenated executive answer text ("" unless mode=="exec_turns")
    mode: str  # "exec_turns" | "no_exec_attribution" | "no_qa_boundary"
    n_exec_turns: int


def isolate_executive_qa(transcript: str) -> ExecQAResult:
    """Return only what executives said during the Q&A portion.

    mode=="exec_turns" is the only success state. "no_qa_boundary" means
    qa_extract found no Q&A start; "no_exec_attribution" means the Q&A was
    found but no turn could be attributed to an executive. Callers must
    treat both failure modes as missing (the plain full-Q&A metrics already
    exist separately) and log their counts — never silently substitute the
    whole transcript.
    """
    transcript = transcript.lstrip("﻿ \n")

    boundary = find_qa_start(transcript)
    if boundary is None:
        return ExecQAResult(text="", mode="no_qa_boundary", n_exec_turns=0)
    qa_text = transcript[boundary:]
    prepared = transcript[:boundary]

    # Executive identity set: roster ∪ prepared-remarks speakers.
    header = prepared[:4000]
    execs = _parse_roster(header, r"Executives?\s*:", [r"Analysts?\s*:", r"Operator\s*:"])
    analysts = _parse_roster(header, r"Analysts?\s*:", [r"Operator\s*:", r"Executives?\s*:"])
    execs |= {spk for spk, _ in _split_turns(prepared)}
    execs.discard("operator")
    execs -= analysts
    exec_surnames = {e.split()[-1] for e in execs if e}
    analyst_surnames = {a.split()[-1] for a in analysts if a}

    exec_chunks = []
    for speaker, words in _split_turns(qa_text):
        if not speaker or speaker == "operator" or speaker in analysts:
            continue
        # full-name match, or surname-only match for label variants
        # ("Bill" vs "William Sullivan") when the surname isn't an analyst's
        if speaker in execs or (
            speaker.split()[-1] in exec_surnames
            and speaker.split()[-1] not in analyst_surnames
        ):
            exec_chunks.append(words)

    if not exec_chunks:
        return ExecQAResult(text="", mode="no_exec_attribution", n_exec_turns=0)
    return ExecQAResult(
        text=" ".join(exec_chunks), mode="exec_turns", n_exec_turns=len(exec_chunks)
    )
