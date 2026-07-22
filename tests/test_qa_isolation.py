from src.qa_isolation import isolate_executive_qa

SYNTHETIC = (
    "﻿ Executives: Jane Doe - CEO John Roe - CFO "
    "Analysts : Amy Wu - BigBank Bob Ray - OtherBank "
    "Operator : Welcome to the Acme earnings call. Jane Doe, you may begin. "
    "Jane Doe : Thank you. Prepared remarks about a great quarter follow now. "
    "John Roe : Financial details and guidance for the year ahead of us all. "
    "Operator : We will now take our first question from Amy Wu of BigBank. "
    "Amy Wu - BigBank: What is your risk outlook for next year? "
    "Jane Doe : We see no material risk at this time. "
    "Bob Ray : A question about margins? "
    "Sam Lee : I am unknown, never spoke in prepared remarks, not rostered. "
    "John Roe : Margins are stable."
)


def test_exec_answers_only():
    r = isolate_executive_qa(SYNTHETIC)
    assert r.mode == "exec_turns"
    # Jane's and John's answers are kept
    assert "no material risk" in r.text
    assert "Margins are stable" in r.text
    # prepared remarks are NOT included
    assert "great quarter" not in r.text
    # analyst questions ("Name - Firm:" labels too) and operator are NOT included
    assert "risk outlook" not in r.text
    assert "Welcome to the Acme" not in r.text
    assert "about margins" not in r.text


def test_unknown_qa_only_speaker_excluded():
    # Sam Lee is neither rostered nor a prepared-remarks speaker -> treated
    # as analyst and excluded (conservative default)
    r = isolate_executive_qa(SYNTHETIC)
    assert "never spoke in prepared remarks" not in r.text


def test_qa_header_variant():
    t = (
        "Alice Smith : Prepared remarks here, going on for a decent while, "
        "covering revenue and product and so on in some depth for the intro. "
        "Question-and-Answer Session Operator : First up is Carl Fox. "
        "Carl Fox : My question? Alice Smith : Our answer is maybe."
    )
    r = isolate_executive_qa(t)
    assert r.mode == "exec_turns"
    assert "Our answer is maybe" in r.text
    assert "My question" not in r.text


def test_no_boundary_returns_no_qa_mode():
    r = isolate_executive_qa("Alice Smith : Just a monologue with no questions section.")
    assert r.mode == "no_qa_boundary"
    assert r.text == ""


def test_attribution_failure_is_reported_not_papered_over():
    # Q&A boundary exists but no turn can be attributed to an executive
    t = (
        "Operator monologue with no named company speakers at all beforehand. "
        "Our first question comes from someone. UNPARSEABLE lowercase blob"
    )
    r = isolate_executive_qa(t)
    assert r.mode == "no_exec_attribution"
    assert r.text == ""
