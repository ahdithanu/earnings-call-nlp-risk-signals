"""Print annotation-candidate windows for benchmark transcripts.

Annotator tooling: shows each transcript's head and tail plus a context
window around every hit of a BROAD Q&A-transition cue set — deliberately
broader than the extractor's own markers, so the annotator's attention is
not anchored to what the extractor can already find. For each cue hit it
also locates the start of the enclosing speaker utterance (the gold
boundary convention — see evals/ANNOTATION_GUIDE.md).

This prints evidence for a human/model reader to judge; it decides
nothing. Usage: python -m evals.show_candidates <id> [<id> ...]
"""

import re
import sys
from pathlib import Path

CACHE = Path("evals/cache")

CUES = re.compile(
    r"question-and-answer|question and answer|questions\s*&(?:amp;)?\s*answers|q\s*&\s*a\b"
    r"|first question|question comes from|question today|question is from"
    r"|line is open|operator instructions|open the (?:floor|line|call)"
    r"|take (?:your|any|our analysts.|the first) questions?|begin the question"
    r"|q[- ]?and[- ]?a session|we will now take|poll for questions",
    re.IGNORECASE,
)
# speaker labels: SA colon style and MF name/--/title style
_SA_LABEL = re.compile(
    r"(?:^|\s)((?:[A-Z][\w.'’\-]*\s*){1,4}(?:-\s*[A-Z][\w.'’&\-]*(?:\s+[A-Z][\w.'’&\-]*){0,4})?\s*:)"
)
_MF_LABEL = re.compile(r"\n[^\n]{2,60}\n--\n[^\n]{2,90}\n")


def utterance_start(text: str, pos: int) -> int | None:
    """Offset of the speaker label that opens the utterance containing pos."""
    best = None
    for m in _MF_LABEL.finditer(text, max(0, pos - 4000), pos + 1):
        best = m.start() + 1  # skip the leading newline
    if best is not None:
        return best
    for m in _SA_LABEL.finditer(text, max(0, pos - 2000), pos + 1):
        if m.start(1) <= pos:
            best = m.start(1)
    return best


def show(tid: str) -> None:
    text = (CACHE / f"{tid}.txt").read_text(encoding="utf-8")
    n = len(text)
    print(f"\n{'=' * 90}\n### {tid}  ({n} chars)\n{'=' * 90}")
    print(f"--- HEAD [0:400] ---\n{text[:400]!r}")
    hits = list(CUES.finditer(text))
    print(f"--- {len(hits)} cue hits ---")
    shown = 0
    last_end = -1
    for m in hits:
        if m.start() < last_end:  # collapse overlapping windows
            continue
        last_end = m.start() + 250
        us = utterance_start(text, m.start())
        shown += 1
        print(
            f"\n[hit {shown}] offset={m.start()} ({m.start() / n:.0%}) cue={m.group(0)!r} "
            f"utterance_start={us}"
        )
        ctx_from = us if us is not None and m.start() - us < 600 else max(0, m.start() - 150)
        print(repr(text[ctx_from : m.start() + 250]))
    print(f"\n--- TAIL [-350:] ---\n{text[-350:]!r}")


if __name__ == "__main__":
    for tid in sys.argv[1:]:
        show(tid)
