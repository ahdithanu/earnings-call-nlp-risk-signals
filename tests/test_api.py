import pytest
from fastapi.testclient import TestClient

from earnings_signals.api import _RAG_CACHE, MAX_TEXT_BYTES, app

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


@pytest.fixture
def rag_index_env(tmp_path, monkeypatch):
    """A tiny hashing-embedder index served via RAG_INDEX_DIR (hermetic)."""
    from earnings_signals.rag import Chunk, HashingEmbedder, HybridIndex

    chunks = [
        Chunk(
            chunk_id="ACME-2025Q1-000",
            ticker="ACME",
            company="Acme Corp",
            sector="Industrials",
            year=2025,
            quarter=1,
            speaker="jane doe",
            text="We see meaningful supply chain disruptions in Vietnam and expect "
            "freight costs to remain elevated through the first half of next year.",
        ),
        Chunk(
            chunk_id="PHRM-2025Q1-000",
            ticker="PHRM",
            company="Pharma Inc",
            sector="Health Care",
            year=2025,
            quarter=1,
            speaker="pat kim",
            text="Our pharmaceutical pipeline advanced with two new clinical trials "
            "this year and enrollment is ahead of the original schedule.",
        ),
    ]
    HybridIndex.build(chunks, HashingEmbedder(), scope="test fixture").save(tmp_path / "idx")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "idx"))
    _RAG_CACHE.clear()
    yield
    _RAG_CACHE.clear()


def test_ask_grounded_answer(rag_index_env):
    r = client.post("/ask", json={"question": "What did they say about supply chain disruptions?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"]
    assert "supply chain disruptions" in body["answer"]
    assert body["citations"][0]["ticker"] == "ACME"
    assert body["index"]["n_chunks"] == 2
    assert body["caveat"].endswith("not investment advice.")


def test_ask_ticker_entitlement_filter(rag_index_env):
    r = client.post(
        "/ask",
        json={"question": "What did they say about supply chains?", "tickers": ["PHRM"]},
    )
    body = r.json()
    # ACME's chunk is masked out before ranking: either PHRM-only citations
    # or an honest refusal, never the out-of-scope document
    tickers = {c["ticker"] for c in body["citations"]}
    assert "ACME" not in tickers


def test_ask_refuses_out_of_kb(rag_index_env):
    r = client.post("/ask", json={"question": "zzqx wvvt plargh fmoo kkjzy"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] and body["answer"] is None


def test_ask_validation_and_missing_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "nowhere"))
    _RAG_CACHE.clear()
    assert client.post("/ask", json={"question": "hi"}).status_code == 422  # too short
    r = client.post("/ask", json={"question": "a valid question?"})
    assert r.status_code == 503
    assert "not built" in r.json()["detail"]
    _RAG_CACHE.clear()


def test_insight_endpoint():
    r = client.get("/insight/NVDA")
    # 200 once data/processed/latest_insights.json is committed; 503 if absent
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert {"headline", "claim", "driver", "character", "terms", "caveat"} <= set(body)
        assert body["ticker"] == "NVDA"
        # lowercase ticker resolves too
        assert client.get("/insight/nvda").status_code == 200
        assert client.get("/insight/NOTATICKER").status_code == 404
