"""Forward-looking signal report: score each ticker's most recent call.

The panel analysis (scripts/analyze_uncertainty_growth.py) establishes that
within-company spikes in Q&A uncertainty density predict weaker next-quarter
TTM EPS growth. This script turns that into a monitoring view: for each
ticker's LATEST available earnings call, how elevated is its Q&A uncertainty
versus that company's own history?

Strictly out-of-sample: the latest quarter is z-scored against the ticker's
PRIOR quarters only, so the signal never uses information from the quarter
being scored. Tickers need MIN_HISTORY prior usable quarters to be scored.

The report is only as fresh as the underlying dataset (currently through
2025Q1 for 13 tickers, 2024Q4 for most). ``quarters_behind`` counts how far
each ticker's latest call lags the newest quarter in the panel — treat
anything > 1 as stale rather than a live signal.

Writes results/latest_uncertainty_signals.csv, most-elevated hedging first.
"""

import pandas as pd

PARQUET = "data/processed/tech_uncertainty_features.parquet"
OUT_CSV = "results/latest_uncertainty_signals.csv"

MIN_QA_TOKENS = 500
MIN_HISTORY = 12


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["qa_isolated"] & (df["total_tokens_qa"] >= MIN_QA_TOKENS)].copy()
    df["density"] = df["uncertainty_density_qa"].astype(float)
    df = df.sort_values(["ticker", "year", "quarter"])

    panel_max_qidx = (df["year"] * 4 + df["quarter"] - 1).max()

    rows = []
    for ticker, g in df.groupby("ticker"):
        hist, latest = g.iloc[:-1], g.iloc[-1]
        if len(hist) < MIN_HISTORY:
            continue
        mu, sd = hist["density"].mean(), hist["density"].std(ddof=0)
        if sd == 0:
            continue
        prev = hist.iloc[-1]
        rows.append({
            "ticker": ticker,
            "company": latest["company"],
            "quarter": latest["datacqtr"],
            "earnings_date": latest["earnings_date"],
            "quarters_behind": int(panel_max_qidx - (latest["year"] * 4 + latest["quarter"] - 1)),
            "n_history": len(hist),
            "density_qa": latest["density"],
            "density_z": (latest["density"] - mu) / sd,
            "density_pctile": (hist["density"] < latest["density"]).mean() * 100,
            "change_vs_prev_call": latest["density"] - prev["density"],
        })

    out = pd.DataFrame(rows).sort_values("density_z", ascending=False)
    out.to_csv(OUT_CSV, index=False)

    fresh = out[out["quarters_behind"] <= 1]
    print(f"scored {len(out)} tickers ({len(fresh)} within 1 quarter of panel edge)")
    print(f"\nmost elevated hedging vs own history (top 10, fresh only):")
    cols = ["ticker", "quarter", "earnings_date", "density_qa", "density_z",
            "density_pctile", "change_vs_prev_call"]
    print(fresh.head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\nmost subdued hedging (bottom 5, fresh only):")
    print(fresh.tail(5)[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
