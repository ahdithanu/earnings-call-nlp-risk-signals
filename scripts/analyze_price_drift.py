"""Does Q&A hedging predict post-call STOCK returns, not just EPS? (v2)

Merges data/processed/price_outcomes.parquet (from scripts/fetch_prices.py)
with the panel's Q&A uncertainty density and runs the SAME fixed-effects
specification as the EPS analysis, with the return as the outcome:

  ret ~ density_z + ticker FE (+ quarter FE), SE clustered by ticker

for both the immediate 5-day reaction and the next-quarter drift. Returns are
winsorized 1/99. Writes results/price_drift.txt.

Run scripts/fetch_prices.py first (needs FMP_API_KEY).
"""

import io
import os

import pandas as pd
import statsmodels.formula.api as smf

PANEL = "data/processed/tech_uncertainty_features.parquet"
PRICES = "data/processed/price_outcomes.parquet"
OUT_TXT = "results/price_drift.txt"
MIN_QA_TOKENS = 500
WINSOR_PCT = 0.01

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def main() -> None:
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["qa_isolated"] & (panel["total_tokens_qa"] >= MIN_QA_TOKENS)]
    dens = panel[["ticker", "datacqtr", "uncertainty_density_qa"]].dropna()
    prices = pd.read_parquet(PRICES)

    df = prices.merge(dens, on=["ticker", "datacqtr"], how="inner")
    df["density_z"] = df.groupby("ticker")["uncertainty_density_qa"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    df = df.dropna(subset=["density_z"])

    emit(f"merged calls with both a price outcome and Q&A density: {len(df)} "
         f"({df['ticker'].nunique()} tickers)")

    for outcome, label in [("ret_5d", "5-day post-call reaction"),
                           ("ret_fwd_q", "next-quarter drift (~63 trading days)")]:
        d = df.dropna(subset=[outcome]).copy()
        lo, hi = d[outcome].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
        d["y"] = d[outcome].clip(lo, hi)
        emit(f"\n=== {label}: {outcome} ~ density_z + FE (SE clustered by ticker) ===")
        emit(f"n = {len(d)}")
        for spec, fe in [("ticker FE", "C(ticker)"),
                         ("ticker + quarter FE", "C(ticker) + C(datacqtr)")]:
            fit = smf.ols(f"y ~ density_z + {fe}", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["ticker"]})
            emit(f"  {spec}: coef on density_z = {fit.params['density_z']:+.3f} % per 1 SD "
                 f"(t={fit.tvalues['density_z']:+.2f}, p={fit.pvalues['density_z']:.3f})")

    os.makedirs("results", exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
