"""Retrieval evaluation: recall@K and MRR for bm25 / dense / hybrid.

Methodology — synthetic query inversion, the standard label-free retrieval
eval: sample chunks from the index (seeded), build a natural-language query
from each chunk's most DISTINCTIVE terms (high tf·idf, stopwords and
corpus-common terms excluded), and measure whether retrieval brings back
the source. Two relevance levels:

  chunk-level — the exact source chunk is in the top K (strict)
  call-level  — any chunk of the source ticker+quarter is in the top K
                (what a user usually needs: the right call)

Plus a refusal check: shuffled-vocabulary junk queries must be refused by
the answer layer, and real queries must not be.

Honest limitation, stated here and in the report: term-based queries favor
BM25 by construction — the query shares exact tokens with its source. The
dense half earns its keep on paraphrase ("worried about demand" vs "softer
order patterns"), which this eval under-measures; fusion's value shows in
dense rescuing queries whose terms are common across the corpus. A
human-labeled query set is the follow-up once real usage exists.

Usage:  python -m scripts.eval_rag [--index data/processed/rag_index] [--n 200]
"""

import argparse
import io
import math
import random
from collections import Counter

from earnings_signals.rag.embedding import get_embedder
from earnings_signals.rag.index import HybridIndex
from earnings_signals.uncertainty import tokenize

OUT_TXT = "results/rag_eval.txt"
KS = (1, 5, 10, 20)
N_JUNK = 25
QUERY_TERMS = 5
MAX_DF_FRAC = 0.05  # terms in >5% of chunks are not distinctive

_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "we",
        "our",
        "us",
        "you",
        "your",
        "they",
        "their",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "he",
        "she",
        "his",
        "her",
        "them",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
        "so",
        "if",
        "then",
        "than",
        "there",
        "here",
        "when",
        "what",
        "which",
        "who",
        "how",
        "why",
        "all",
        "any",
        "some",
        "more",
        "most",
        "other",
        "into",
        "over",
        "under",
        "about",
        "just",
        "also",
        "very",
        "really",
        "know",
        "think",
        "going",
        "go",
        "get",
        "got",
        "lot",
        "bit",
        "year",
        "quarter",
        "years",
        "quarters",
        "time",
        "one",
        "two",
        "three",
        "first",
        "second",
        "half",
        "kind",
        "sort",
        "things",
        "thing",
        "way",
        "well",
        "right",
        "now",
        "today",
        "look",
        "looking",
        "said",
        "say",
        "see",
        "seen",
        "continue",
        "terms",
    ]
)

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def make_query(chunk_tokens: list[str], company: str, df: dict[str, int], n_docs: int) -> str:
    tf = Counter(chunk_tokens)
    scored = [
        (tf[t] * math.log(n_docs / (1 + df.get(t, 0))), t)
        for t in tf
        if t not in _STOP and len(t) > 2 and df.get(t, 0) <= MAX_DF_FRAC * n_docs
    ]
    terms = [t for _, t in sorted(scored, reverse=True)[:QUERY_TERMS]]
    return f"What did {company} executives say about {' '.join(terms)}?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/processed/rag_index")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    index = HybridIndex.load(args.index)
    embedder = get_embedder(index.embedder_name)
    n_docs = len(index.chunks)
    df = {term: len(plist) for term, plist in index._bm25.postings.items()}

    rng = random.Random(args.seed)
    # sample only substantive chunks so queries have enough distinctive terms
    eligible = [i for i, c in enumerate(index.chunks) if len(c.text.split()) >= 40]
    sample = rng.sample(eligible, min(args.n, len(eligible)))
    queries = [
        (i, make_query(tokenize(index.chunks[i].text), index.chunks[i].company, df, n_docs))
        for i in sample
    ]

    emit("=== retrieval eval: synthetic query inversion ===")
    emit(f"index: {index.scope} — {n_docs} chunks, embedder {index.embedder_name}")
    emit(f"queries: {len(queries)} (seed {args.seed}), relevance = source chunk / source call")
    emit(f"example query: {queries[0][1]!r}")
    emit("")
    emit(f"{'mode':<8} {'level':<6} " + " ".join(f"recall@{k:<3}" for k in KS) + "  MRR")

    for mode in ("bm25", "dense", "hybrid"):
        hits_chunk = {k: 0 for k in KS}
        hits_call = {k: 0 for k in KS}
        rr_sum = 0.0
        for src_i, query in queries:
            src = index.chunks[src_i]
            results = index.search(query, embedder, k=max(KS), mode=mode)
            ids = [r.chunk.chunk_id for r in results]
            calls = [(r.chunk.ticker, r.chunk.year, r.chunk.quarter) for r in results]
            src_call = (src.ticker, src.year, src.quarter)
            for k in KS:
                hits_chunk[k] += src.chunk_id in ids[:k]
                hits_call[k] += src_call in calls[:k]
            if src.chunk_id in ids:
                rr_sum += 1.0 / (ids.index(src.chunk_id) + 1)
        n = len(queries)
        emit(
            f"{mode:<8} {'chunk':<6} "
            + " ".join(f"{hits_chunk[k] / n:>8.1%} " for k in KS)
            + f" {rr_sum / n:.3f}"
        )
        emit(f"{mode:<8} {'call':<6} " + " ".join(f"{hits_call[k] / n:>8.1%} " for k in KS))

    # --- refusal sanity -------------------------------------------------------
    from earnings_signals.rag.answer import answer_question

    vocab = [t for t in df if t not in _STOP and len(t) > 3]
    junk_refused = 0
    for _ in range(N_JUNK):
        scrambled = ["".join(rng.sample(list(w), len(w))) for w in rng.sample(vocab, 6)]
        if answer_question(" ".join(scrambled), index, embedder).refused:
            junk_refused += 1
    real_answered = sum(not answer_question(q, index, embedder).refused for _, q in queries[:50])
    emit("")
    emit("=== refusal check (answer layer) ===")
    emit(f"junk queries refused: {junk_refused}/{N_JUNK}")
    emit(f"real queries answered: {real_answered}/50")
    emit("")
    emit(
        "caveat: term-inversion queries share exact tokens with their source, "
        "which favors bm25; dense retrieval earns its keep on paraphrase, "
        "under-measured here. Fusion (hybrid) should match bm25 on these "
        "queries while adding robustness — that, not beating bm25, is the bar."
    )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
