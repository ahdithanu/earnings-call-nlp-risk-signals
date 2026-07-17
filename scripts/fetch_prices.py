"""Fetch post-earnings-call price drift for the tech panel (v2: price outcomes).

For every earnings call in data/processed/tech_uncertainty_features.parquet,
pulls the company's daily closes from Financial Modeling Prep and computes the
post-call return over two horizons (src/price_drift.py). Writes
data/processed/price_outcomes.parquet.

Requires an FMP API key in the FMP_API_KEY environment variable. This runs
where there is both a key and open network egress — e.g. the GitHub Actions
workflow .github/workflows/price-outcomes.yml with an FMP_API_KEY repo secret
— NOT the restricted sandbox. The drift math itself is unit-tested
(tests/test_price_drift.py); only the network fetch is environment-dependent.
"""

import os
import sys
import time

import pandas as pd
import requests

from src.price_drift import drift_outcomes

PANEL = "data/processed/tech_uncertainty_features.parquet"
OUT = "data/processed/price_outcomes.parquet"
BASE = "https://financialmodelingprep.com/stable/historical-price-eod/light"
FROM_DATE = "2013-01-01"


def fetch_closes(ticker: str, api_key: str, to_date: str) -> tuple[list[str], list[float]]:
    """Ascending (dates, closes) for one ticker, or ([], []) on failure."""
    r = requests.get(BASE, params={"symbol": ticker, "from": FROM_DATE,
                                   "to": to_date, "apikey": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        return [], []
    rows = sorted(data, key=lambda d: d["date"])  # API returns newest-first
    dates = [d["date"] for d in rows]
    closes = [float(d["price"]) for d in rows]
    return dates, closes


def main() -> None:
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        sys.exit("FMP_API_KEY is not set — add it as an env var / GitHub Actions "
                 "secret. The FMP historical-price endpoint requires it.")

    panel = pd.read_parquet(PANEL)
    calls = panel.dropna(subset=["earnings_date"])[
        ["ticker", "datacqtr", "year", "quarter", "earnings_date"]
    ].drop_duplicates()
    to_date = str(pd.Timestamp.utcnow().date() + pd.Timedelta(days=1))

    rows, ok, miss = [], 0, 0
    for ticker, g in calls.groupby("ticker"):
        try:
            dates, closes = fetch_closes(ticker, api_key, to_date)
        except Exception as e:  # network / API hiccup: skip the ticker, keep going
            print(f"  {ticker}: fetch failed ({e}); skipping")
            miss += 1
            continue
        if not dates:
            print(f"  {ticker}: no price data")
            miss += 1
            continue
        ok += 1
        for r in g.itertuples():
            out = drift_outcomes(dates, closes, r.earnings_date)
            rows.append({
                "ticker": ticker, "datacqtr": r.datacqtr,
                "year": int(r.year), "quarter": int(r.quarter),
                "earnings_date": r.earnings_date, **out,
            })
        time.sleep(0.2)  # be polite to the API

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    print(f"tickers fetched: {ok} ok, {miss} missing")
    print(f"wrote {OUT}: {len(out_df)} calls "
          f"({out_df['ret_fwd_q'].notna().sum()} with next-quarter drift)")


if __name__ == "__main__":
    main()
