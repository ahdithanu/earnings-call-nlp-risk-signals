"""Split executive Q&A speech by the speaker's role (CEO / CFO / IR / other).

The speech turns carry no titles, so roles come from two header sources,
tried in order:
  1. the ``Executives: Name - Title`` roster — present in ~90% of the
     2013-2018 transcripts and essentially absent from 2019 on (the
     dataset's transcript source changed format);
  2. the IR intro prose of the later format ("joining me today are Jane
     Doe, our Chief Executive Officer, and John Roe, our CFO"), parsed by
     ``intro_titles``.
The feature pipeline records which source attributed a row (role_source)
and callers get None when neither works — partial coverage is reported,
never papered over.

Turn speakers are matched to attributed names exactly first, then by
surname (a roster may say "D. Cook" while turns say "Tim Cook"); a surname
carrying two different roles is treated as unmatchable rather than guessed.
A junk (name, title) pair from prose parsing is harmless unless it collides
with a real Q&A speaker's surname, which the ambiguity rule also catches.
"""

import re

from src.qa_isolation import _norm, executive_qa_turns

# Title words that regex name-capture sometimes swallows ("Founder and CEO
# Ken Xie" -> "Founder"; "Executive Chairman" -> "Chairman"). Stripped from
# candidate names; a candidate with nothing left is rejected. Validated
# against the 2018/2019 boundary-continuity check.
_NAME_STOPWORDS = frozenset({
    "founder", "chairman", "chairwoman", "officer", "president", "chief",
    "executive", "vice", "senior", "director", "general", "counsel", "head",
    "investor", "relations", "co", "interim", "group", "corporate",
})


def _clean_name(name: str) -> str | None:
    """Drop title stopwords from a candidate name; None if nothing remains.

    Trailing single-letter tokens are possessive debris ("Ken Xie's" ->
    "ken xie s"), not initials — leading initials ("D. Cook") are kept.
    """
    tokens = [t for t in name.split() if t not in _NAME_STOPWORDS]
    while tokens and len(tokens[-1]) == 1:
        tokens.pop()
    return " ".join(tokens) if tokens else None

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
        name = _clean_name(_norm(mm.group(1)))
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


# Title phrases the intro-prose parser anchors on. Only the roles we score
# (plus IR, which helps disambiguation) — prose is too noisy for a long tail.
_INTRO_TITLE_RE = re.compile(
    r"(?:president\s+and\s+)?chief\s+executive\s+officer|\bceo\b"
    r"|(?:executive\s+vice\s+president\s+and\s+)?chief\s+financial\s+officer|\bcfo\b"
    r"|investor\s+relations",
    re.IGNORECASE,
)
_INTRO_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{1,2}}"
# name-first: "... Jane Doe, our Chief Executive Officer" — a name, a comma,
# then at most a few filler words before the title phrase
_NAME_BEFORE_RE = re.compile(
    rf"({_INTRO_NAME})\s*,\s*(?:[\w'’\-]+\s+){{0,4}}$", re.UNICODE
)
# title-first: "... our CEO, Jane Doe" / "CEO Jane Doe"
_NAME_AFTER_RE = re.compile(rf"^\s*,?\s*({_INTRO_NAME})", re.UNICODE)


def intro_titles(transcript: str) -> list[tuple[str, str]]:
    """(normalized name, title phrase) pairs parsed from the IR intro prose.

    Used for transcripts with no ``Executives:`` roster (the 2019+ format).
    Anchors on CEO/CFO/IR title phrases in the first ~4,000 characters and
    takes the adjacent capitalized name — either "Name, our <title>" or
    "<title>[,] Name". Historical mentions ("then-CEO Steve Jobs") can slip
    through; they are harmless downstream because bucketing only ever
    applies to speakers already attributed as executives of THIS call.
    """
    head = transcript.lstrip("﻿ \n")[:4000]
    pairs = []
    for m in _INTRO_TITLE_RE.finditer(head):
        title = m.group(0)
        before = head[max(0, m.start() - 80):m.start()]
        after = head[m.end():m.end() + 80]
        mb = _NAME_BEFORE_RE.search(before)
        ma = _NAME_AFTER_RE.match(after)
        raw = mb.group(1) if mb else (ma.group(1) if ma else None)
        name = _clean_name(_norm(raw)) if raw else None
        if name:
            pairs.append((name, title))
    return pairs


def exec_qa_by_role(transcript: str) -> tuple[dict[str, str] | None, str | None]:
    """Executive Q&A text grouped by role: (buckets, source).

    buckets is {"ceo": ..., "cfo": ..., "ir": ..., "other": ...} (empty
    string for a role that never speaks) and source is "roster" or "intro"
    depending on which header form attributed the roles. (None, None) when
    neither yields (name, title) pairs or executive turns couldn't be
    isolated at all — callers must treat that as missing, not as zero.
    """
    pairs = roster_titles(transcript)
    source = "roster"
    if not pairs:
        pairs = intro_titles(transcript)
        source = "intro"
    if not pairs:
        return None, None
    turns, mode = executive_qa_turns(transcript)
    if mode != "exec_turns":
        return None, None

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
    return {role: " ".join(chunks) for role, chunks in buckets.items()}, source
