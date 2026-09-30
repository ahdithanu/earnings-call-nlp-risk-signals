"""Fetch post-earnings-call price drift for the S&P 500 panel (v2: price outcomes).

Pulls each panel ticker's daily closes and computes the post-call return over
two horizons (earnings_signals/price_drift.py) into
data/processed/price_outcomes.parquet.

The provider is pluggable (earnings_signals/price_providers.py):
- FMP_API_KEY set          -> Financial Modeling Prep (bulk; one run does the
                              whole panel in minutes).
- ALPHAVANTAGE_API_KEY set -> Alpha Vantage (use when that is your key). The
                              free tier is ~25 requests/day, so runs are
                              RESUMABLE: each run skips tickers already in the
                              parquet and stops cleanly when the daily limit is
                              hit, so coverage accumulates over several days.
PRICE_PROVIDER (auto/fmp/alphavantage, default auto) forces the choice;
PRICE_MAX_TICKERS caps tickers per run (0 = no cap).

Runs where there is a key and open egress — the price-outcomes GitHub Actions
workflow — not the restricted sandbox. The drift math and the provider parsing
are unit-tested; only the live network call is environment-dependent.
"""

import os
import sys
import time
from typing import Any

import pandas as pd
import requests

from earnings_signals.price_drift import drift_outcomes
from earnings_signals.price_providers import (
    ALPHAVANTAGE_URL,
    FMP_URL,
    FROM_DATE,
    PremiumRequired,
    RateLimited,
    parse_alphavantage,
    parse_fmp,
    resolve_provider,
)

PANEL = "data/processed/sp500_uncertainty_features.parquet"
OUT = "data/processed/price_outcomes.parquet"
SLEEP_S = {"fmp": 0.2, "alphavantage": 1.0}


def fetch_fmp(ticker: str, api_key: str, to_date: str) -> tuple[list[str], list[float]]:
    """Ascending (dates, closes) for one ticker from FMP."""
    r = requests.get(
        FMP_URL,
        params={"symbol": ticker, "from": FROM_DATE, "to": to_date, "apikey": api_key},
        timeout=30,
    )
    r.raise_for_status()
    return parse_fmp(r.json())


def fetch_alphavantage(ticker: str, api_key: str) -> tuple[list[str], list[float]]:
    """Ascending (dates, closes) for one ticker from Alpha Vantage — the
    split/dividend-adjusted series, falling back to the free unadjusted daily
    series when the account lacks the premium endpoint."""

    def _call(function: str) -> Any:
        r = requests.get(
            ALPHAVANTAGE_URL,
            params={
                "function": function,
                "symbol": ticker,
                "outputsize": "full",
                "apikey": api_key,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    try:
        return parse_alphavantage(_call("TIME_SERIES_DAILY_ADJUSTED"))
    except PremiumRequired:
        return parse_alphavantage(_call("TIME_SERIES_DAILY"))


def main() -> None:
    fmp_key = os.environ.get("FMP_API_KEY")
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    try:
        provider = resolve_provider(
            os.environ.get("PRICE_PROVIDER", "auto"), bool(fmp_key), bool(av_key)
        )
    except ValueError as e:
        sys.exit(str(e))
    max_tickers = int(os.environ.get("PRICE_MAX_TICKERS", "0"))  # 0 = no cap
    print(f"provider: {provider}")

    panel = pd.read_parquet(PANEL)
    calls = panel.dropna(subset=["earnings_date"])[
        ["ticker", "datacqtr", "year", "quarter", "earnings_date"]
    ].drop_duplicates()
    to_date = str(pd.Timestamp.utcnow().date() + pd.Timedelta(days=1))

    # Resume: skip tickers already fetched (matters for the AV free-tier daily
    # cap — coverage builds up across runs instead of restarting each time).
    existing = pd.read_parquet(OUT) if os.path.exists(OUT) else None
    done = set(existing["ticker"].unique()) if existing is not None else set()
    if done:
        print(f"resuming: {len(done)} tickers already in {OUT}")

    todo = [t for t in calls["ticker"].unique() if t not in done]
    if max_tickers:
        todo = todo[:max_tickers]

    rows: list[dict] = []
    ok = miss = 0
    stopped = False
    for ticker in todo:
        try:
            if provider == "fmp":
                dates, closes = fetch_fmp(ticker, fmp_key, to_date)
            else:
                dates, closes = fetch_alphavantage(ticker, av_key)
        except RateLimited as e:
            print(f"rate limit reached at {ticker}: {e}")
            stopped = True
            break
        except Exception as e:  # network hiccup on one ticker: skip, keep going
            print(f"  {ticker}: fetch failed ({e}); skipping")
            miss += 1
            continue
        if not dates:
            print(f"  {ticker}: no price data")
            miss += 1
            continue
        ok += 1
        for r in calls[calls["ticker"] == ticker].itertuples():
            out = drift_outcomes(dates, closes, r.earnings_date)
            rows.append(
                {
                    "ticker": ticker,
                    "datacqtr": r.datacqtr,
                    "year": int(r.year),
                    "quarter": int(r.quarter),
                    "earnings_date": r.earnings_date,
                    **out,
                }
            )
        time.sleep(SLEEP_S.get(provider, 0.5))

    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    if combined.empty:
        sys.exit("no price rows fetched — check the key/provider and rate limits")
    combined = combined.drop_duplicates(["ticker", "datacqtr"], keep="last")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    combined.to_parquet(OUT, index=False)

    covered = combined["ticker"].nunique()
    total = calls["ticker"].nunique()
    print(
        f"tickers fetched this run: {ok} ok, {miss} missing"
        + (" (stopped on rate limit)" if stopped else "")
    )
    print(
        f"wrote {OUT}: {len(combined)} calls, {covered}/{total} tickers covered "
        f"({combined['ret_fwd_q'].notna().sum()} with next-quarter drift)"
    )
    if covered < total:
        print(
            f"incomplete: {total - covered} tickers remain — rerun to continue "
            "(free Alpha Vantage is ~25/day, so full coverage takes several days)"
        )


if __name__ == "__main__":
    main()
