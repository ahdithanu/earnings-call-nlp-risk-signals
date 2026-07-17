"""Robustness extension: does the hedging→EPS finding generalize beyond tech?

The headline result (scripts/analyze_uncertainty_growth.py) is estimated on
large-cap tech. This script re-runs the SAME specification across ALL 11 GICS
sectors of glopardo/sp500-earnings-transcripts and reports, per sector,
whether elevated Q&A uncertainty predicts weaker next-quarter EPS growth.

It is analysis-only: it does not touch the tech panel, the parquet, or the
explorer. The pipeline is identical to the main analysis — Q&A isolation,
negation-aware uncertainty density, gap-guarded forward TTM EPS growth,
within-ticker z-scored density, ticker fixed effects, ticker-clustered SEs,
growth winsorized 1/99 (here per sector, since EPS-growth scale differs by
sector), Q&A >= 500 tokens.

Writes results/cross_sector_robustness.txt and results/cross_sector.csv.

Requires huggingface.co on first run (cached after). Scores the full ~20k
transcript set, so it is slower than the tech-only build.
"""

import io
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from datasets import load_dataset

from src.features import add_next_quarter_eps
from src.lexicon import load_uncertainty_terms
from src.qa_extract import extract_qa
from src.uncertainty import count_uncertainty

DATASET = "glopardo/sp500-earnings-transcripts"
OUT_TXT = "results/cross_sector_robustness.txt"
OUT_CSV = "results/cross_sector.csv"
MIN_QA_TOKENS = 500
MIN_SECTOR_OBS = 150  # skip sectors too thin to estimate a ticker-FE panel
WINSOR_PCT = 0.01

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def score_row(text: str, lexicon: set[str]) -> float | None:
    """Q&A uncertainty density for one transcript, or None if unusable."""
    qa = extract_qa(text)
    if qa is None:
        return None
    r = count_uncertainty(qa, lexicon)
    if r.total_tokens < MIN_QA_TOKENS:
        return None
    return r.density


def ticker_fe_fit(g: pd.DataFrame):
    """Ticker-FE regression of winsorized growth on within-ticker density_z."""
    return smf.ols("growth_w ~ density_z + C(ticker)", data=g).fit(
        cov_type="cluster", cov_kwds={"groups": g["ticker"]}
    )


def main() -> None:
    lexicon = load_uncertainty_terms()
    df = load_dataset(DATASET)["train"].to_pandas()
    df = df.dropna(subset=["year", "quarter", "sector"]).copy()
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)

    # one row per (ticker, year, quarter): keep the longest transcript
    df["_tlen"] = df["transcript"].str.len()
    df = (df.sort_values("_tlen", ascending=False)
            .drop_duplicates(["ticker", "year", "quarter"]).drop(columns="_tlen"))

    emit(f"scoring {len(df)} transcripts across {df['sector'].nunique()} sectors…")
    df["density"] = df["transcript"].apply(lambda t: score_row(t, lexicon))
    df = df.dropna(subset=["density"]).copy()

    # forward TTM EPS growth (same construction as the tech panel)
    df["eps"] = df["eps12mtrailing_eoq"]
    df = add_next_quarter_eps(df)
    df = df.rename(columns={"eps_growth_next_q": "growth"}).dropna(subset=["growth"])
    emit(f"usable company-quarters (Q&A>= {MIN_QA_TOKENS} tokens, growth present): {len(df)}")

    rows = []
    emit("\n=== per-sector: ticker-FE panel, SE clustered by ticker ===")
    emit(f"{'sector':<26}{'n':>6}{'tickers':>9}{'beta_pp':>10}{'t':>7}{'p':>8}")
    for sector, g in df.groupby("sector"):
        g = g.copy()
        lo, hi = g["growth"].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
        g["growth_w"] = g["growth"].clip(lo, hi)
        g["density_z"] = g.groupby("ticker")["density"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0))
        g = g.dropna(subset=["density_z"])
        if len(g) < MIN_SECTOR_OBS or g["ticker"].nunique() < 5:
            continue
        fit = ticker_fe_fit(g)
        b = fit.params["density_z"]; t = fit.tvalues["density_z"]; p = fit.pvalues["density_z"]
        rows.append({"sector": sector, "n": int(len(g)), "tickers": int(g["ticker"].nunique()),
                     "beta_pp": round(float(b), 3), "t": round(float(t), 2), "p": round(float(p), 4)})
        emit(f"{sector:<26}{len(g):>6}{g['ticker'].nunique():>9}{b:>+10.3f}{t:>+7.2f}{p:>8.3f}")

    # pooled across all sectors: ticker + quarter FE (ticker FE absorbs sector)
    d = df.copy()
    lo, hi = d["growth"].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
    d["growth_w"] = d["growth"].clip(lo, hi)
    d["density_z"] = d.groupby("ticker")["density"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    d = d.dropna(subset=["density_z"])
    pooled = smf.ols("growth_w ~ density_z + C(ticker) + C(datacqtr)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    emit("\n=== pooled, ALL sectors (ticker + quarter FE, clustered by ticker) ===")
    emit(f"coef on density_z = {pooled.params['density_z']:+.3f} pp per 1 SD "
         f"(t={pooled.tvalues['density_z']:+.2f}, p={pooled.pvalues['density_z']:.3f}, "
         f"n={int(pooled.nobs)}, {d['ticker'].nunique()} tickers, "
         f"{d['sector'].nunique()} sectors)")

    neg = sum(1 for r in rows if r["beta_pp"] < 0)
    sig = sum(1 for r in rows if r["p"] < 0.05 and r["beta_pp"] < 0)
    emit(f"\nsectors with a negative coefficient: {neg}/{len(rows)} | "
         f"negative AND p<0.05: {sig}/{len(rows)}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_TXT} and {OUT_CSV}")


if __name__ == "__main__":
    main()
