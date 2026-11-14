from earnings_signals.qa_extract import extract_qa, find_qa_start

# Padding so markers land inside/outside the plausible-position window.
REMARKS = "Revenue grew across all segments this quarter. " * 40


def test_header_marks_qa_start():
    text = REMARKS + "Question-and-Answer Session Operator: Our first question. " + REMARKS
    start = find_qa_start(text)
    assert start is not None
    assert text[start:].startswith("Question-and-Answer Session")


def test_intro_announcement_is_ignored():
    # The intro mentions the Q&A session before any remarks; only the
    # mid-call handoff should be picked up.
    text = (
        "Operator: Later we will conduct a question-and-answer session. "
        + REMARKS
        + "Operator: Your first question comes from Jane Doe. Great quarter. "
        + REMARKS[: len(REMARKS) // 2]
    )
    start = find_qa_start(text)
    assert start is not None
    assert text[start:].lower().startswith("first question")


def test_no_marker_returns_none():
    assert extract_qa(REMARKS) is None
    assert extract_qa("") is None


def test_marker_in_tail_is_ignored():
    # A closing "first question" mention in the last 5% is not a Q&A start.
    text = REMARKS * 5 + "thanks for the first question"
    assert find_qa_start(text) is None


def test_operator_handoff_fallback():
    # No explicit header and no "first question" cue — only the operator's
    # analyst handoff marks the Q&A. The fallback should catch it.
    text = REMARKS + "Our next question comes from Jane Doe of Big Bank. " + REMARKS
    start = find_qa_start(text)
    assert start is not None
    assert text[start:].lower().startswith("next question comes from")


def test_header_still_wins_over_handoff():
    # When both a header and a later handoff exist, the header (higher
    # priority) must win — the fallback never overrides it.
    text = (
        REMARKS
        + "Question-and-Answer Session. "
        + "Our next question comes from Jane Doe of Big Bank. "
        + REMARKS
    )
    start = find_qa_start(text)
    assert text[start:].startswith("Question-and-Answer Session")


def test_extract_returns_suffix():
    text = REMARKS + "Question and Answer Session begins now. " + REMARKS
    qa = extract_qa(text)
    assert qa is not None
    assert qa == text[text.index("Question and Answer") :]


# Handoff phrasings recovered by the evals/ benchmark's dev split (see
# evals/results/): operators saying "first call", "we'll go [first] to
# <Name> with <Firm>", and an IR host's "now take questions from".
def test_handoff_variants_from_benchmark():
    variants = [
        "And our first call comes from Matt Summerville with D.A. Davidson. ",
        "The first call is to Amit Daryanani with Evercore. ",
        "We'll go to the line of Ric Prentiss with Raymond James. ",
        "We'll go first to A.J. Rice with UBS. ",
        "And we'll go to Eric Coldwell with Baird. ",
        "We'll now take questions from the analyst community. ",
    ]
    for v in variants:
        text = REMARKS + v + REMARKS
        start = find_qa_start(text)
        assert start is not None, v
        assert len(REMARKS) <= start < len(REMARKS) + len(v), v  # boundary inside the handoff


def test_go_to_requires_capitalized_name():
    # "we'll go to" followed by a lowercase noun phrase is prepared-remarks
    # language, not an analyst handoff — it must not create a boundary.
    for s in [
        "We'll go to market with the new product this year. ",
        "We'll go to slide five of the presentation deck now. ",
        "We'll go first to review the segment financials today. ",
    ]:
        assert find_qa_start(REMARKS + s + REMARKS) is None, s
