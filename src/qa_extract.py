"""Q&A-section isolation for glopardo/sp500-earnings-transcripts transcripts.

Transcripts in this dataset are a single string: a speaker-prefixed flow of
prepared remarks followed by the analyst Q&A. There is no structural field
separating the two, so the boundary is found by text markers, validated
against the Information Technology subset (2,919 transcripts):

- An explicit "Question-and-Answer Session" header exists in only ~61% of
  transcripts, and usually ALSO appears in the call intro ("we will conduct
  a question-and-answer session"), so a match only counts when it falls at
  a plausible position inside the call (past the intro, before the tail).
- The operator's first-question handoff ("your first question comes from")
  is the fallback marker; together the two cover ~96% of tech transcripts.
- When neither marker is found at a plausible position, the caller should
  fall back to the full transcript and record that Q&A isolation failed —
  never silently score prepared remarks as if they were Q&A.
"""

import re

_QA_HEADER_RE = re.compile(
    r"question-and-answer session|question and answer session|question-and-answer period",
    re.IGNORECASE,
)
_FIRST_QUESTION_RE = re.compile(
    r"first question(?: today)?(?: comes| is| will come)?",
    re.IGNORECASE,
)

# A marker only marks the Q&A start when it sits past the call intro (where
# the Q&A session is merely announced) and before the very end of the call.
# On the tech subset the accepted split lands at 23-60% of the transcript
# (p5-p95), comfortably inside this window.
MIN_POSITION = 0.15
MAX_POSITION = 0.95


def find_qa_start(text: str) -> int | None:
    """Return the character offset where the Q&A section starts, else None.

    The explicit section header is preferred; the operator's first-question
    handoff is the fallback. Matches outside the plausible-position window
    are ignored (intro announcements, closing remarks).
    """
    n = len(text)
    if n == 0:
        return None
    for pattern in (_QA_HEADER_RE, _FIRST_QUESTION_RE):
        for m in pattern.finditer(text):
            if MIN_POSITION * n <= m.start() <= MAX_POSITION * n:
                return m.start()
    return None


def extract_qa(text: str) -> str | None:
    """Return the Q&A section of ``text``, or None when no boundary is found."""
    start = find_qa_start(text)
    if start is None:
        return None
    return text[start:]
