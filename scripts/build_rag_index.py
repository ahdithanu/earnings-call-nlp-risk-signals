"""Build the retrieval index behind POST /ask — merged two-source corpus.

Mirrors the project's two-layer ingestion split (see
scripts/fetch_recent_signals.py):

  panel source   glopardo/sp500-earnings-transcripts — the validated
                 backbone, ends at the panel watermark (2025Q1)
  live source    Rogersurf/earnings-call-transcripts — Motley Fool
                 scrapes through the current earnings season; only
                 quarters STRICTLY NEWER than each ticker's last panel
                 quarter are taken, so the sources never double-count

Scope: the most recent N quarters across the merged corpus (default 6,
spanning the watermark). The full 2013+ history would be millions of
chunk-embeddings; recent calls are what "what are executives saying"
questions are for. The artifact records its scope in meta.json.

The index is a BUILD ARTIFACT, not a committed file: the weekly
refresh-signals workflow rebuilds it and uploads it to the rolling
`rag-index` GitHub release (fetch with scripts/fetch_rag_index.py).

Usage:  python -m scripts.build_rag_index [--quarters 6] [--out DIR]
                                          [--embedder potion-base-8M]
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from datasets import load_dataset

from earnings_signals.rag.chunking import Chunk, chunk_transcript
from earnings_signals.rag.embedding import DEFAULT_MODEL, get_embedder
from earnings_signals.rag.index import HybridIndex
from earnings_signals.universe import select_universe

PANEL_DATASET = "glopardo/sp500-earnings-transcripts"
RECENT_DATASET = "Rogersurf/earnings-call-transcripts"
DEFAULT_OUT = "data/processed/rag_index"
MIN_TRANSCRIPT_CHARS = 2000


def load_panel_calls() -> pd.DataFrame:
    """Validated-source calls: ticker/company/sector/year/quarter/transcript."""
    df = load_dataset(PANEL_DATASET)["train"].to_pandas()
    df = df[select_universe(df)].dropna(subset=["year", "quarter"]).copy()
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df["_tlen"] = df["transcript"].str.len()
    df = df.sort_values("_tlen", ascending=False).drop_duplicates(["ticker", "year", "quarter"])
    df["source"] = "panel"
    return df[["ticker", "company", "sector", "year", "quarter", "transcript", "source"]]


def load_recent_calls(panel: pd.DataFrame) -> pd.DataFrame:
    """Live-source calls strictly newer than each ticker's panel watermark.

    Same conventions as scripts/fetch_recent_signals.py: calendar quarter
    from call_date, longest transcript per (ticker, quarter). Sector comes
    from the panel (the live source has none); company from the live rows.
    """
    sector_map = panel.drop_duplicates("ticker").set_index("ticker")["sector"]
    last_panel = (panel["year"] * 4 + panel["quarter"] - 1).groupby(panel["ticker"]).max()

    raw = load_dataset(RECENT_DATASET)["train"].to_pandas()
    df = raw[raw["ticker"].isin(set(panel["ticker"]))].copy()
    df["cd"] = pd.to_datetime(df["call_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["cd"])
    df["year"] = df["cd"].dt.year
    df["quarter"] = df["cd"].dt.quarter
    # the cleaned text parses best; fall back to the raw scrape when absent
    clean = df["transcript_clean"]
    df["transcript"] = clean.where(
        clean.str.len().fillna(0) >= MIN_TRANSCRIPT_CHARS, df["transcript"]
    )
    df = df.dropna(subset=["transcript"])
    df["_tlen"] = df["transcript"].str.len()
    df = df.sort_values("_tlen", ascending=False).drop_duplicates(["ticker", "year", "quarter"])

    qidx = df["year"] * 4 + df["quarter"] - 1
    df = df[qidx > df["ticker"].map(last_panel)].copy()
    df["sector"] = df["ticker"].map(sector_map)
    df["source"] = "rogersurf"
    return df[["ticker", "company", "sector", "year", "quarter", "transcript", "source"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarters", type=int, default=6, help="most recent N quarters to index")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--embedder", default=DEFAULT_MODEL.split("/")[-1])
    args = parser.parse_args()

    panel = load_panel_calls()
    recent = load_recent_calls(panel)
    merged = pd.concat([panel, recent], ignore_index=True)

    merged["_qidx"] = merged["year"] * 4 + merged["quarter"]
    cutoff = int(merged["_qidx"].max()) - args.quarters + 1
    scoped = merged[merged["_qidx"] >= cutoff].sort_values(["ticker", "year", "quarter"])
    q_lo = f"{cutoff // 4 if cutoff % 4 else cutoff // 4 - 1}Q{cutoff % 4 or 4}"
    scope = f"executive Q&A answers, last {args.quarters} quarters (from {q_lo})"
    by_src = scoped["source"].value_counts().to_dict()
    print(
        f"scope: {scope} — {len(scoped)} calls, {scoped['ticker'].nunique()} tickers "
        f"(panel {by_src.get('panel', 0)}, live {by_src.get('rogersurf', 0)})"
    )

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
            source=str(row.source),
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
