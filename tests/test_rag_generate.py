from types import SimpleNamespace

import pytest

import earnings_signals.rag.generate as gen
from earnings_signals.rag import Chunk, HashingEmbedder, HybridIndex
from earnings_signals.rag.generate import check_grounded, generate_answer

CHUNKS = [
    Chunk(
        chunk_id="ACME-2026Q1-000",
        ticker="ACME",
        company="Acme Corp",
        sector="Industrials",
        year=2026,
        quarter=1,
        speaker="jane doe",
        text="We see meaningful supply chain disruptions in Vietnam and expect "
        "freight costs to remain elevated through the first half of next year.",
    ),
    Chunk(
        chunk_id="ACME-2026Q1-001",
        ticker="ACME",
        company="Acme Corp",
        sector="Industrials",
        year=2026,
        quarter=1,
        speaker="john roe",
        text="Cloud revenue grew forty percent this quarter and we expect that "
        "momentum to continue as enterprise demand stays strong across regions.",
    ),
]

# every sentence cites both excerpts, so the text is grounded regardless
# of which chunk retrieval ranks first
GROUNDED = (
    "Acme expects supply chain disruptions in Vietnam with freight costs "
    "staying elevated [1][2]. Cloud revenue grew forty percent with strong "
    "enterprise demand [1][2]."
)
FABRICATED = "The company announced a quantum computing partnership on Mars [1]."


@pytest.fixture
def index_and_embedder():
    emb = HashingEmbedder()
    return HybridIndex.build(CHUNKS, emb, scope="test"), emb


def kept(index, emb):
    return index.search("supply chain and cloud revenue", emb, k=2)


# deterministic "kept" list for the pure check_grounded tests: [1] is always
# the supply-chain chunk, [2] the cloud chunk
FIXED_KEPT = [SimpleNamespace(chunk=c, score=1.0) for c in CHUNKS]


def test_check_grounded_accepts_supported_cited_text():
    ok, why = check_grounded(
        "Acme expects supply chain disruptions in Vietnam with freight costs staying elevated [1].",
        FIXED_KEPT,
    )
    assert ok, why


def test_check_grounded_rejects_bad_marker():
    ok, why = check_grounded("Supply chains are disrupted [7].", FIXED_KEPT)
    assert not ok and "nonexistent" in why


def test_check_grounded_rejects_uncited_sentence():
    text = "Freight costs stay elevated [1]. Separately, margins will expand a lot."
    ok, why = check_grounded(text, FIXED_KEPT)
    assert not ok and "uncited" in why


def test_check_grounded_rejects_fabrication():
    ok, why = check_grounded(FABRICATED, FIXED_KEPT)
    assert not ok and "lexical support" in why


def test_check_grounded_rejects_citation_free_answer():
    ok, why = check_grounded("Supply chains are disrupted.", FIXED_KEPT)
    assert not ok and "no citations" in why


def test_generate_answer_happy_path(index_and_embedder, monkeypatch):
    index, emb = index_and_embedder
    monkeypatch.setattr(gen, "_call_model", lambda q, kept, model: GROUNDED)
    ans = generate_answer("What did Acme say about supply chains?", index, emb)
    assert ans.mode == "generative" and not ans.refused
    assert ans.answer == GROUNDED
    # both excerpts were cited, so both appear as citations
    assert {c.chunk_id for c in ans.citations} == {c.chunk_id for c in CHUNKS}


def test_generate_answer_model_side_refusal(index_and_embedder, monkeypatch):
    index, emb = index_and_embedder
    monkeypatch.setattr(gen, "_call_model", lambda q, kept, model: gen.NOT_IN_CONTEXT)
    ans = generate_answer("What did Acme say about supply chains?", index, emb)
    assert ans.refused and ans.mode == "generative"
    assert "model-side" in (ans.reason or "")


def test_generate_answer_falls_back_on_ungrounded_output(index_and_embedder, monkeypatch):
    index, emb = index_and_embedder
    monkeypatch.setattr(gen, "_call_model", lambda q, kept, model: FABRICATED)
    ans = generate_answer("What did Acme say about supply chains?", index, emb)
    assert ans.mode == "extractive_fallback" and not ans.refused
    assert "groundedness check" in (ans.fallback_reason or "")
    # the served answer is verbatim corpus text, not the fabricated output
    corpus = " ".join(c.text for c in CHUNKS)
    assert all(line.split('"')[1] in corpus for line in (ans.answer or "").splitlines())


def test_generate_answer_falls_back_on_api_error(index_and_embedder, monkeypatch):
    index, emb = index_and_embedder

    def boom(q, kept, model):
        raise RuntimeError("api down")

    monkeypatch.setattr(gen, "_call_model", boom)
    ans = generate_answer("What did Acme say about supply chains?", index, emb)
    assert ans.mode == "extractive_fallback"
    assert "generation failed" in (ans.fallback_reason or "")


def test_generate_answer_keeps_retrieval_refusal(index_and_embedder, monkeypatch):
    index, emb = index_and_embedder

    def never(q, kept, model):  # retrieval refuses before any model call
        raise AssertionError("model should not be called")

    monkeypatch.setattr(gen, "_call_model", never)
    ans = generate_answer("zzqx wvvt plargh fmoo kkjzy", index, emb)
    assert ans.refused and ans.mode == "generative"
