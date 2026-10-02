"""Build the retrieval index behind POST /ask.

Chunks executive Q&A answers (earnings_signals/rag/chunking.py) from the
S&P 500 corpus and embeds them into a hybrid index artifact at
data/processed/rag_index/.

Scope: the most recent N quarters present in the dataset (default 4). The
full 2013–2025 corpus would be millions of chunk-embeddings and hundreds
of MB — more than the repo or the free-tier service should carry — and
recent calls are what Q&A over "what are executives saying" is for. Widen
with --quarters for a local full-history index; the artifact records its
scope in meta.json either way.

Usage:  python -m scripts.build_rag_index [--quarters 4] [--out DIR]
                                          [--embedder potion-base-8M]
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset

from earnings_signals.rag.chunking import Chunk, chunk_transcript
from earnings_signals.rag.embedding import DEFAULT_MODEL, get_embedder
from earnings_signals.rag.index import HybridIndex
from earnings_signals.universe import select_universe

DATASET = "glopardo/sp500-earnings-transcripts"
DEFAULT_OUT = "data/processed/rag_index"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarters", type=int, default=4, help="most recent N quarters to index")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--embedder", default=DEFAULT_MODEL.split("/")[-1])
    args = parser.parse_args()

    df = load_dataset(DATASET)["train"].to_pandas()
    df = df[select_universe(df)].dropna(subset=["year", "quarter"]).copy()
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df["_tlen"] = df["transcript"].str.len()
    df = df.sort_values("_tlen", ascending=False).drop_duplicates(["ticker", "year", "quarter"])

    df["_qidx"] = df["year"] * 4 + df["quarter"]
    cutoff = int(df["_qidx"].max()) - args.quarters + 1
    scoped = df[df["_qidx"] >= cutoff].sort_values(["ticker", "year", "quarter"])
    q_lo = f"{cutoff // 4 if cutoff % 4 else cutoff // 4 - 1}Q{cutoff % 4 or 4}"
    scope = f"executive Q&A answers, last {args.quarters} quarters (from {q_lo})"
    print(f"scope: {scope} — {len(scoped)} calls, {scoped['ticker'].nunique()} tickers")

    chunks: list[Chunk] = []
    no_chunks = 0
    for row in scoped.itertuples():
        call_chunks = chunk_transcript(
            row.transcript,
            ticker=str(row.ticker),
            company=str(row.company),
            sector=str(row.sector),
            year=int(row.year),
            quarter=int(row.quarter),
        )
        if not call_chunks:
            no_chunks += 1
        chunks.extend(call_chunks)
    print(
        f"chunks: {len(chunks)} from {len(scoped) - no_chunks} calls "
        f"({no_chunks} calls yielded none: no Q&A boundary / no exec attribution)"
    )

    embedder = get_embedder(args.embedder)
    t0 = time.time()
    index = HybridIndex.build(chunks, embedder, scope=scope)
    print(f"embedded with {embedder.name} in {time.time() - t0:.1f}s (dim={embedder.dim})")

    index.save(args.out)
    total = sum(f.stat().st_size for f in Path(args.out).iterdir())
    for f in sorted(Path(args.out).iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    print(f"wrote {args.out}: {len(chunks)} chunks, {total / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
