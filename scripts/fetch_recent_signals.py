"""Extend the explorer with recent quarters (2025Q2+) from a live source.

The validated panel (data/processed/tech_uncertainty_features.parquet, from
glopardo/sp500-earnings-transcripts) ends at 2025Q1. This script appends
newer company-quarters scored from Rogersurf/earnings-call-transcripts,
which runs through 2026, so the interactive explorer shows a current
signal instead of a static history.

IMPORTANT — this output feeds the EXPLORER ONLY. It never touches the
panel parquet, and the regression/analysis scripts remain single-source.
Two reasons the recent rows are kept separate:

  1. Different transcript source (Rogersurf scrapes Motley Fool) than the
     panel (Seeking Alpha), so Q&A uncertainty density sits on a slightly
     different scale — Rogersurf runs ~0.075 lower. Bolting it on raw would
     bias the within-company signal downward and manufacture a fake "hedging
     fell in 2025-2026" story. We CALIBRATE instead: fit an affine map
     (panel_density ~ a + b * rogersurf_density) on the company-quarters
     present in BOTH sources, then apply it. The two sources correlate 0.94
     on the overlap, so this is a scale alignment, not a fudge; the fit
     coefficients and overlap size are logged every run.
  2. Recent quarters have no realized next-quarter EPS yet, so they carry
     the density signal only (eps columns NaN) and render as density bars
     without an EPS point — the same treatment the panel's newest quarter
     already gets.

Only quarters strictly newer than each ticker's last panel quarter are
appended, so there is no cross-source double-counting. 5 of the 72 tickers
are absent from Rogersurf (ANSS, JNPR, MPWR, MSI, TXN) and simply keep
their existing end date.

Writes data/processed/recent_uncertainty_signals.parquet.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.lexicon import load_uncertainty_terms
from src.qa_extract import extract_qa
from src.uncertainty import count_uncertainty

PANEL = "data/processed/tech_uncertainty_features.parquet"
RECENT_DATASET = "Rogersurf/earnings-call-transcripts"
OUT = "data/processed/recent_uncertainty_signals.parquet"
MIN_QA_TOKENS = 500


def score(text: str, lexicon: set[str]) -> tuple[float, int] | None:
    """Calibrated-input density for one transcript's Q&A, or None."""
    if not isinstance(text, str):
        return None
    qa = extract_qa(text)
    if qa is None:
        return None
    r = count_uncertainty(qa, lexicon)
    if r.total_tokens < MIN_QA_TOKENS:
        return None
    return r.density, r.total_tokens


def main() -> None:
    lexicon = load_uncertainty_terms()
    panel = pd.read_parquet(PANEL)
    universe = set(panel["ticker"].unique())
    # last calendar-quarter index present per ticker in the panel
    panel_qidx = (panel["year"] * 4 + panel["quarter"] - 1)
    last_panel = panel_qidx.groupby(panel["ticker"]).max()

    raw = load_dataset(RECENT_DATASET)["train"].to_pandas()
    df = raw[raw["ticker"].isin(universe)].copy()
    df["cd"] = pd.to_datetime(df["call_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["cd"])
    df["year"] = df["cd"].dt.year
    df["quarter"] = df["cd"].dt.quarter  # calendar quarter, matches panel
    df["qidx"] = df["year"] * 4 + df["quarter"] - 1

    # one row per (ticker, calendar quarter); keep the longest transcript
    df["_len"] = df["transcript"].str.len()
    df = (
        df.sort_values("_len", ascending=False)
        .drop_duplicates(["ticker", "year", "quarter"])
        .reset_index(drop=True)
    )

    scored = df["transcript"].apply(lambda t: score(t, lexicon))
    df = df[scored.notna()].copy()
    df["dens_raw"] = [s[0] for s in scored[scored.notna()]]
    df["total_tokens_qa"] = [s[1] for s in scored[scored.notna()]]
    df["datacqtr"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)

    # ---- calibration on the cross-source overlap -------------------------
    panel_key = panel[["ticker", "datacqtr", "uncertainty_density_qa"]].dropna()
    overlap = df.merge(panel_key, on=["ticker", "datacqtr"])
    b, a = np.polyfit(overlap["dens_raw"], overlap["uncertainty_density_qa"], 1)
    r = np.corrcoef(overlap["dens_raw"], overlap["uncertainty_density_qa"])[0, 1]
    print(f"calibration overlap: {len(overlap)} company-quarters, corr={r:.3f}")
    print(f"panel_density ~ {a:+.4f} + {b:.4f} * rogersurf_density")
    df["uncertainty_density_qa"] = a + b * df["dens_raw"]

    # ---- keep only quarters strictly newer than the panel ----------------
    df["last_panel_qidx"] = df["ticker"].map(last_panel)
    recent = df[df["qidx"] > df["last_panel_qidx"]].copy()
    recent = recent.sort_values(["ticker", "year", "quarter"])

    out = pd.DataFrame({
        "ticker": recent["ticker"],
        "company": recent["company"],
        "year": recent["year"].astype(int),
        "quarter": recent["quarter"].astype(int),
        "datacqtr": recent["datacqtr"],
        "earnings_date": recent["cd"].dt.strftime("%Y-%m-%d"),
        "qa_isolated": True,
        "total_tokens_qa": recent["total_tokens_qa"].astype(int),
        "uncertainty_density_qa": recent["uncertainty_density_qa"].astype(float),
        "source": "rogersurf",
    })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out)} recent company-quarters, "
          f"{out['ticker'].nunique()} tickers, "
          f"{out['datacqtr'].min()}–{out['datacqtr'].max()}")
    print("newest quarter counts:")
    print(out["datacqtr"].value_counts().sort_index().tail(6).to_string())


if __name__ == "__main__":
    main()
