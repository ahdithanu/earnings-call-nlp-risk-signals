import numpy as np
import pytest

from earnings_signals.rag import (
    Chunk,
    HashingEmbedder,
    HybridIndex,
    answer_question,
    chunk_transcript,
)
from earnings_signals.rag.chunking import MAX_CHUNK_TOKENS

# A synthetic call in the corpus's "Name : words" format: roster, prepared
# remarks (long enough to put the Q&A boundary inside the position window),
# then attributable executive answers.
FILLER = "Prepared remarks continue with many details about the quarter. " * 12
SYNTHETIC = (
    "Executives: Jane Doe - CEO John Roe - CFO "
    "Analysts : Amy Wu - BigBank "
    f"Operator : Welcome to the Acme earnings call. Jane Doe : Thank you. {FILLER} "
    "Operator : We will now take our first question from Amy Wu of BigBank. "
    "Amy Wu - BigBank : What risks do you see to margins next year? "
    "Jane Doe : We see meaningful supply chain disruptions in Vietnam and expect "
    "freight costs to remain elevated through the first half of next year overall. "
    "Amy Wu - BigBank : And on the cloud side? "
    "John Roe : Cloud revenue grew forty percent this quarter and we expect that "
    "momentum to continue as enterprise migration demand stays very strong indeed. "
    "Jane Doe : Thanks. "
    "Operator : That concludes the call."
)


def make_chunks():
    return chunk_transcript(
        SYNTHETIC, ticker="ACME", company="Acme Corp", sector="Industrials", year=2025, quarter=1
    )


def other_chunk(seq=0, text="Our pharmaceutical pipeline advanced with two new trials this year."):
    return Chunk(
        chunk_id=f"PHRM-2025Q1-{seq:03d}",
        ticker="PHRM",
        company="Pharma Inc",
        sector="Health Care",
        year=2025,
        quarter=1,
        speaker="pat kim",
        text=text,
    )


def build_index(extra=None):
    chunks = make_chunks() + (extra or [other_chunk()])
    emb = HashingEmbedder()
    return HybridIndex.build(chunks, emb, scope="test"), emb


def test_chunking_turns_with_metadata():
    chunks = make_chunks()
    # two substantive exec answers; "Thanks." and short turns dropped
    assert len(chunks) == 2
    assert {c.speaker for c in chunks} == {"jane doe", "john roe"}
    assert all(c.ticker == "ACME" and c.quarter_label == "2025Q1" for c in chunks)
    assert "supply chain disruptions" in chunks[0].text
    assert "Cloud revenue" in chunks[1].text
    assert chunks[0].chunk_id == "ACME-2025Q1-000"


def test_chunking_splits_long_turns():
    long_turn = " ".join(
        f"Sentence number {i} talks about the outlook in detail." for i in range(60)
    )
    t = SYNTHETIC.replace(
        "We see meaningful supply chain disruptions in Vietnam and expect "
        "freight costs to remain elevated through the first half of next year overall.",
        long_turn,
    )
    chunks = chunk_transcript(
        t, ticker="ACME", company="Acme Corp", sector="Industrials", year=2025, quarter=1
    )
    jane = [c for c in chunks if c.speaker == "jane doe"]
    assert len(jane) > 1
    assert all(len(c.text.split()) <= MAX_CHUNK_TOKENS for c in jane)


def test_chunking_empty_on_unparseable_call():
    assert (
        chunk_transcript(
            "no structure here at all", ticker="X", company="X", sector="?", year=2025, quarter=1
        )
        == []
    )


# A synthetic call in the Motley Fool format (the live Rogersurf source):
# "Name\n--\nTitle" turn headers, a Questions & Answers section marker,
# analyst turns identified by their title.
MF_BODY = "We keep executing on the plan and see solid momentum everywhere. " * 2
MF_SYNTHETIC = (
    "Prepared Remarks:\n"
    "\nOperator\n--\n\nGood morning and welcome to the Acme earnings call.\n"
    f"\nJane Doe\n--\nChief Executive Officer\n\n{MF_BODY}\n"
    f"\nJohn Roe\n--\nChief Financial Officer\n\n{MF_BODY}\n"
    "Questions & Answers:\n"
    "\nAmy Wu\n--\nBigBank -- Analyst\n\nWhat risks do you see to margins next year?\n"
    "\nJane Doe\n--\nChief Executive Officer\n\nWe see meaningful supply chain "
    "disruptions in Vietnam and expect freight costs to remain elevated through "
    "the first half of next year overall.\n"
    "\nBob Ray\n--\nOtherBank -- Analyst\n\nAnd on the cloud side?\n"
    "\nJohn Roe\n--\nChief Financial Officer\n\nCloud revenue grew forty percent "
    "this quarter and we expect that momentum to continue as enterprise demand "
    "stays very strong across every region we serve today.\n"
)


def test_chunking_motley_fool_format():
    chunks = chunk_transcript(
        MF_SYNTHETIC,
        ticker="ACME",
        company="Acme Corp",
        sector="Industrials",
        year=2026,
        quarter=1,
        source="rogersurf",
    )
    # only the two Q&A executive answers: prepared remarks are before the
    # marker, analyst turns are excluded by title
    assert len(chunks) == 2
    assert {c.speaker for c in chunks} == {"jane doe", "john roe"}
    assert "supply chain disruptions" in chunks[0].text
    assert "margins next year" not in " ".join(c.text for c in chunks)  # analyst text out
    assert all(c.source == "rogersurf" and c.quarter_label == "2026Q1" for c in chunks)


def test_hybrid_search_finds_exact_terms():
    index, emb = build_index()
    for mode in ("hybrid", "bm25", "dense"):
        top = index.search("supply chain disruptions in Vietnam", emb, k=1, mode=mode)[0]
        assert top.chunk.speaker == "jane doe", mode


def test_ticker_filter_masks_before_ranking():
    index, emb = build_index()
    results = index.search("supply chain disruptions", emb, k=5, tickers=["PHRM"])
    assert results and all(r.chunk.ticker == "PHRM" for r in results)
    assert index.search("anything", emb, k=5, tickers=["NOPE"]) == []


def test_save_load_roundtrip(tmp_path):
    index, emb = build_index()
    index.save(tmp_path / "idx")
    loaded = HybridIndex.load(tmp_path / "idx")
    assert loaded.embedder_name == emb.name
    assert len(loaded.chunks) == len(index.chunks)
    a = index.search("cloud revenue momentum", emb, k=2)
    b = loaded.search("cloud revenue momentum", emb, k=2)
    assert [r.chunk.chunk_id for r in a] == [r.chunk.chunk_id for r in b]
    # fp16 round-trip keeps scores close
    assert a[0].dense_score == pytest.approx(b[0].dense_score, abs=1e-2)


def test_embedder_mismatch_rejected():
    index, _ = build_index()
    with pytest.raises(ValueError, match="built with"):
        index.search("x", HashingEmbedder(dim=64), k=1)


def test_grounded_answer_is_verbatim_with_citations():
    index, emb = build_index()
    ans = answer_question("What did executives say about cloud revenue growth?", index, emb)
    assert not ans.refused and ans.answer is not None
    corpus = " ".join(c.text for c in index.chunks)
    for line in ans.answer.splitlines():
        quoted = line.split('"')[1]
        assert quoted in corpus  # structural grounding: verbatim only
    assert ans.citations and ans.citations[0].ticker in {"ACME", "PHRM"}
    assert ans.caveat.endswith("not investment advice.")


def test_out_of_kb_question_is_refused():
    index, emb = build_index()
    ans = answer_question("zzqx wvvt plargh fmoo kkjzy", index, emb)
    assert ans.refused and ans.answer is None
    assert "does not match the knowledge base" in (ans.reason or "")


def test_empty_scope_is_refused():
    index, emb = build_index()
    ans = answer_question("cloud revenue", index, emb, tickers=["NOPE"])
    assert ans.refused and "scope" in (ans.reason or "")


def test_hashing_embedder_deterministic_and_normalized():
    emb = HashingEmbedder()
    v1 = emb.encode(["cloud revenue grew", "unrelated pharma trials"])
    v2 = emb.encode(["cloud revenue grew", "unrelated pharma trials"])
    assert np.allclose(v1, v2)
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)
    sim_same = float(v1[0] @ emb.encode(["cloud revenue growth"])[0])
    sim_diff = float(v1[0] @ v1[1])
    assert sim_same > sim_diff
