from src.qa_extract import extract_qa, find_qa_start

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


def test_extract_returns_suffix():
    text = REMARKS + "Question and Answer Session begins now. " + REMARKS
    qa = extract_qa(text)
    assert qa is not None
    assert qa == text[text.index("Question and Answer") :]
