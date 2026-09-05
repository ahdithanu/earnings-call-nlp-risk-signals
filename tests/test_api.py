from fastapi.testclient import TestClient

from earnings_signals.api import MAX_TEXT_BYTES, app

client = TestClient(app)

# A miniature call with roster, prepared remarks, and Q&A, so every scope
# (full/qa/execqa/ceo/cfo) is derivable.
TRANSCRIPT = (
    "Executives: Jane Doe - CEO John Roe - CFO "
    "Analysts : Amy Wu - BigBank "
    "Operator : Welcome to the Acme earnings call today everyone. "
    "Jane Doe : Prepared remarks about a fine quarter, thank you all. "
    "John Roe : Financial details and the guidance for the year ahead. "
    "Operator : We will now take our first question from Amy Wu. "
    "Amy Wu : What is your risk outlook for the next fiscal year? "
    "Jane Doe : There is significant uncertainty and risk ahead of us. "
    "John Roe : Margins may possibly vary, but we see no material risk."
)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["lexicon_terms"] == 297


def test_score_all_scopes_derivable():
    r = client.post("/score", json={"text": TRANSCRIPT})
    assert r.status_code == 200
    body = r.json()
    assert set(body["scopes"]) == {"full", "qa", "execqa", "ceo", "cfo"}
    assert body["qa_isolated"] and body["exec_isolated"]
    assert body["role_source"] == "roster"
    ceo = body["scopes"]["ceo"]
    # "uncertainty" and "risk" are LM uncertainty terms ("significant" is not)
    assert ceo["uncertainty_count"] == 2
    # CFO's "no material risk" is negation-excluded; "may"/"possibly"/"vary" count
    cfo = body["scopes"]["cfo"]
    assert cfo["negation_excluded"] >= 1
    assert cfo["uncertainty_count"] >= 3
    assert set(ceo["tone_density"]) == {"negative", "positive", "litigious", "constraining"}


def test_score_plain_text_degrades_to_full_scope_only():
    r = client.post("/score", json={"text": "We may possibly see some risk."})
    assert r.status_code == 200
    body = r.json()
    assert "full" in body["scopes"]
    assert not body["qa_isolated"]
    assert body["scopes"]["full"]["uncertainty_count"] == 3


def test_score_rejects_empty_and_oversized():
    assert client.post("/score", json={"text": ""}).status_code == 422
    big = "word " * (MAX_TEXT_BYTES // 4)
    assert client.post("/score", json={"text": big}).status_code == 413


def test_signals_endpoint():
    r = client.get("/signals?limit=5")
    # 200 with rows when the committed watchlist exists; 503 only if absent
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        rows = r.json()
        assert 0 < len(rows) <= 5
        assert {"ticker", "density_z", "quarter"} <= set(rows[0])
