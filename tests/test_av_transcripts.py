from datetime import UTC, datetime

from earnings_signals.rag.av import exec_qa_turns_from_av, parse_av_response
from earnings_signals.rag.chunking import chunks_from_turns
from scripts.fetch_av_transcripts import quarter_labels, select_targets

TURNS = [
    {"speaker": "Operator", "title": "Operator", "content": "Welcome to the call."},
    {"speaker": "Jane Doe", "title": "Chief Executive Officer", "content": "Prepared remarks."},
    {"speaker": "Amy Wu", "title": "Analyst", "content": "What risks do you see ahead?"},
    {
        "speaker": "Jane Doe",
        "title": "Chief Executive Officer",
        "content": "We see meaningful supply chain disruptions in Vietnam and expect "
        "freight costs to remain elevated through the first half of next year.",
        "sentiment": "0.2",
    },
    {"speaker": "Operator", "title": "Operator", "content": "Next question please."},
    {
        "speaker": "John Roe",
        "title": "Executive Vice President, Chief Financial Officer",
        "content": "Cloud revenue grew forty percent this quarter and we expect that "
        "momentum to continue as enterprise demand stays strong across every region.",
    },
]


def test_parse_av_response_shapes():
    ok = parse_av_response({"symbol": "ACME", "quarter": "2026Q3", "transcript": TURNS})
    assert ok.status == "ok" and len(ok.turns) == len(TURNS)
    assert parse_av_response({"transcript": []}).status == "empty"
    assert parse_av_response({}).status == "empty"
    rate = parse_av_response({"Information": "Our standard API rate limit is 25 requests per day."})
    assert rate.status == "rate_limited"
    err = parse_av_response({"Error Message": "Invalid API call."})
    assert err.status == "error" and "Invalid" in err.detail
    assert parse_av_response(["not", "a", "dict"]).status == "error"


def test_exec_qa_turns_filtering():
    turns = exec_qa_turns_from_av(TURNS)
    # Q&A starts at the analyst turn; prepared remarks, analyst and operator
    # turns are excluded; both exec answers kept with normalized speakers
    assert [s for s, _ in turns] == ["jane doe", "john roe"]
    assert "supply chain disruptions" in turns[0][1]
    # no analyst-titled turn -> no Q&A boundary -> nothing
    assert exec_qa_turns_from_av(TURNS[:2]) == []


def test_av_turns_chunk_end_to_end():
    chunks = chunks_from_turns(
        exec_qa_turns_from_av(TURNS),
        ticker="ACME",
        company="Acme Corp",
        sector="Industrials",
        year=2026,
        quarter=3,
        source="alphavantage",
    )
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "ACME-2026Q3-000"
    assert all(c.source == "alphavantage" for c in chunks)


def test_quarter_labels_cross_year():
    assert quarter_labels(datetime(2026, 2, 10, tzinfo=UTC)) == ["2026Q1", "2025Q4"]
    assert quarter_labels(datetime(2026, 10, 7, tzinfo=UTC)) == ["2026Q4", "2026Q3"]


def test_select_targets_priorities_and_retry():
    now = datetime(2026, 10, 7, tzinfo=UTC)
    cache = {
        ("AAA", "2026Q4"): {"status": "ok", "fetched": "2026-10-01"},  # done: never again
        ("BBB", "2026Q4"): {"status": "empty", "fetched": "2026-09-01"},  # stale: retry
        ("CCC", "2026Q4"): {"status": "empty", "fetched": "2026-10-06"},  # fresh miss: wait
    }
    targets = select_targets(["AAA", "BBB", "CCC"], ["2026Q4"], cache, now)
    assert ("AAA", "2026Q4") not in targets
    assert ("CCC", "2026Q4") not in targets
    assert targets == [("BBB", "2026Q4")]
    # never-attempted pairs come before stale retries
    targets = select_targets(["AAA", "BBB", "DDD"], ["2026Q4"], cache, now)
    assert targets == [("DDD", "2026Q4"), ("BBB", "2026Q4")]
