from src.exec_roles import exec_qa_by_role, role_of_title, roster_titles

TRANSCRIPT = (
    "﻿ Executives: Jane Doe - President and CEO John Roe - SVP, CFO "
    "Pat Kim - VP, Investor Relations Sam Lee - President, Widgets Group "
    "Analysts : Amy Wu - BigBank "
    "Operator : Welcome to the Acme call. Pat Kim, you may begin. "
    "Pat Kim : Thanks. Here are the usual disclaimers before we start today. "
    "Jane Doe : Prepared remarks about the great quarter we finished now. "
    "John Roe : Financial details and guidance for the year ahead of us. "
    "Operator : We will now take our first question from Amy Wu of BigBank. "
    "Amy Wu - BigBank: What is your risk outlook for the next fiscal year? "
    "Jane Doe : Strategy answer from the chief executive right here. "
    "John Roe : Numbers answer from the finance chief right here. "
    "Amy Wu - BigBank: A follow-up? "
    "Sam Lee : Divisional answer from a business unit president here."
)


def test_role_of_title_buckets():
    assert role_of_title("President and CEO") == "ceo"
    assert role_of_title("SVP, CFO") == "cfo"
    assert role_of_title("Senior Vice President and Chief Financial Officer") == "cfo"
    # finance-led combination resolves to the finance role
    assert role_of_title("CFO and interim CEO") == "cfo"
    assert role_of_title("VP, Investor Relations") == "ir"
    assert role_of_title("President, Chemical Analysis Group") == "other"


def test_roster_titles_pairs():
    pairs = dict(roster_titles(TRANSCRIPT))
    assert pairs["jane doe"].startswith("President and CEO")
    assert pairs["john roe"].startswith("SVP, CFO")
    # "Relations" may bleed into the next entry's name (documented parser
    # limitation); the prefix is stable and still buckets to IR
    assert pairs["pat kim"].startswith("VP, Investor")


def test_split_by_role():
    by_role = exec_qa_by_role(TRANSCRIPT)
    assert by_role is not None
    assert "Strategy answer" in by_role["ceo"]
    assert "Numbers answer" in by_role["cfo"]
    # prepared remarks stay out of every bucket
    assert "great quarter" not in by_role["ceo"]
    assert "Financial details" not in by_role["cfo"]
    # analyst text is not in any bucket
    assert all("risk outlook" not in text for text in by_role.values())
    # a rostered non-C-level exec (BU president) falls into "other"
    assert "Divisional answer" in by_role["other"]


def test_no_roster_returns_none():
    t = (
        "Alice Smith : Prepared remarks going on for a while here in the "
        "opening stretch of this particular call transcript for the test. "
        "Operator : We will now take our first question from Carl Fox. "
        "Carl Fox : Question? Alice Smith : Answer text."
    )
    assert exec_qa_by_role(t) is None
