"""Split executive Q&A speech by the speaker's role (CEO / CFO / IR / other).

Role information exists only in the ``Executives:`` roster that ~40% of
transcripts open with ("Didier Hirsch - SVP, CFO"); the speech turns
themselves carry no titles. So this view is inherently partial: callers get
None when no roster is parseable, and the feature pipeline records that in
a ``role_attributed`` flag rather than pretending full coverage.

Turn speakers are matched to roster names exactly first, then by surname
(roster may say "D. Cook" while turns say "Tim Cook"); a surname shared by
two rostered executives with different roles is treated as unmatchable
rather than guessed.
"""

import re

from src.qa_isolation import _norm, executive_qa_turns

ROLE_CEO = "ceo"
ROLE_CFO = "cfo"
ROLE_IR = "ir"
ROLE_OTHER = "other"
ROLES = (ROLE_CEO, ROLE_CFO, ROLE_IR, ROLE_OTHER)

# Roster entries are "Name - Title Name - Title ..." with no delimiter
# between entries; names are found as capitalized runs before a dash, and
# each title runs until the next name-dash. A name token must contain a
# lowercase letter (or be an initial like "D."), so title acronyms directly
# before a name ("...SVP, CFO Pat Kim - ...") aren't swallowed into it;
# "McMullen"-style internal capitals still qualify. Titles that end in
# Title-Case words ("President, Chemical Analysis Group Fred Strohmeier")
# can still bleed into the captured name — downstream matching therefore
# falls back to the surname, the token right before the dash, which is
# reliable.
_NAME_TOKEN = r"(?:[A-Z](?=[\w'’\-]*[a-zà-ÿ])[\w'’\-]*|[A-Z]\.)"
_NAME_DASH_RE = re.compile(
    rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})\s*[-–—]", re.UNICODE
)


def roster_titles(transcript: str) -> list[tuple[str, str]]:
    """(normalized name, raw title) pairs from the 'Executives:' block."""
    head = transcript.lstrip("﻿ \n")[:4000]
    m = re.search(r"Executives?\s*:", head, re.IGNORECASE)
    if not m:
        return []
    block = head[m.end():]
    for lbl in (r"Analysts?\s*:", r"Operator\s*:"):
        e = re.search(lbl, block, re.IGNORECASE)
        if e:
            block = block[:e.start()]
    matches = list(_NAME_DASH_RE.finditer(block))
    pairs = []
    for i, mm in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        title = block[mm.end():end].strip(" ,;")
        name = _norm(mm.group(1))
        if name:
            pairs.append((name, title))
    return pairs


def role_of_title(title: str) -> str:
    """Bucket a raw roster title into CEO / CFO / IR / other.

    Finer buckets (CTO, sales, product...) are too rare on earnings calls
    to support a view; combined titles ("President and CEO") bucket to the
    C-level match. CFO is checked before CEO so finance-led combinations
    resolve to the finance role.
    """
    t = title.lower()
    if re.search(r"chief financial|\bcfo\b", t):
        return ROLE_CFO
    if re.search(r"chief executive|\bceo\b", t):
        return ROLE_CEO
    # "\binvestor\b" alone also matches titles truncated by roster parsing
    # ("VP, Investor" when "Relations <Next Name>" bled into the next entry)
    if re.search(r"investor relations|\binvestor\b", t):
        return ROLE_IR
    return ROLE_OTHER


def exec_qa_by_role(transcript: str) -> dict[str, str] | None:
    """Executive Q&A text grouped by role, or None when unattributable.

    Returns {"ceo": ..., "cfo": ..., "ir": ..., "other": ...} (empty string
    for a role that never speaks). None when the transcript has no parseable
    roster or executive turns couldn't be isolated at all — callers must
    treat that as missing, not as zero.
    """
    pairs = roster_titles(transcript)
    if not pairs:
        return None
    turns, mode = executive_qa_turns(transcript)
    if mode != "exec_turns":
        return None

    full_name_role = {}
    surname_role: dict[str, str] = {}
    ambiguous_surnames = set()
    for name, title in pairs:
        role = role_of_title(title)
        full_name_role[name] = role
        surname = name.split()[-1]
        if surname in surname_role and surname_role[surname] != role:
            ambiguous_surnames.add(surname)
        surname_role[surname] = role

    buckets: dict[str, list[str]] = {r: [] for r in ROLES}
    for speaker, words in turns:
        role = full_name_role.get(speaker)
        if role is None:
            surname = speaker.split()[-1]
            if surname not in ambiguous_surnames:
                role = surname_role.get(surname)
        # an exec turn with no roster match (e.g. exec absent from a stale
        # roster) still counted as executive speech -> "other"
        buckets[role or ROLE_OTHER].append(words)
    return {role: " ".join(chunks) for role, chunks in buckets.items()}
