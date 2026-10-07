"""Pluggable historical-price providers for the post-call drift fetch.

Two providers behind one interface (ticker -> ascending ``(dates, closes)``):

- **Financial Modeling Prep** (``fmp``): bulk-friendly — one plan pulls the
  whole S&P 500 in minutes, split-adjusted closes.
- **Alpha Vantage** (``alphavantage``): useful when that is the key you hold.
  The free tier is capped at ~25 requests/day and returns UNADJUSTED closes
  (a split inside a return window distorts it); the adjusted series needs a
  premium plan, so the fetcher asks for adjusted and falls back to unadjusted.

The drift math (``price_drift.py``) is provider-agnostic; only the fetch and
parse differ. This module holds the pure, unit-tested pieces — response
parsing and provider selection — with no network dependency; the thin HTTP
call lives in scripts/fetch_prices.py (network stays out of the typed package).
"""

from typing import Any

FROM_DATE = "2013-01-01"
FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/light"
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

PROVIDERS = ("fmp", "alphavantage")


class RateLimited(Exception):
    """A provider signalled its rate limit; the caller stops and keeps whatever
    it has fetched so far (Alpha Vantage's free tier is ~25 requests/day)."""


class PremiumRequired(Exception):
    """The requested Alpha Vantage function needs a premium plan; the caller
    falls back to the free (unadjusted) daily series."""


def _to_series(pairs: list[tuple[str, float]]) -> tuple[list[str], list[float]]:
    """Sort (date, close) pairs ascending and split into aligned lists.

    ISO date strings sort chronologically, so a plain string sort is correct.
    """
    pairs.sort()
    return [d for d, _ in pairs], [c for _, c in pairs]


def parse_fmp(payload: Any) -> tuple[list[str], list[float]]:
    """(dates, closes) from an FMP ``historical-price-eod/light`` payload."""
    if not isinstance(payload, list) or not payload:
        return [], []
    return _to_series([(row["date"], float(row["price"])) for row in payload])


def parse_alphavantage(payload: Any) -> tuple[list[str], list[float]]:
    """(dates, closes) from an Alpha Vantage ``TIME_SERIES_DAILY[_ADJUSTED]``
    payload, preferring the adjusted close.

    Alpha Vantage returns throttle and premium notices as plain-text messages
    under ``Note``/``Information`` rather than an HTTP error. The daily-cap
    message mentions *premium* too, so rate-limit indicators are checked first.
    """
    if not isinstance(payload, dict):
        return [], []
    note = payload.get("Note") or payload.get("Information")
    if isinstance(note, str):
        low = note.lower()
        rate_markers = ("rate limit", "call frequency", "requests per day", "25 requests")
        if any(marker in low for marker in rate_markers):
            raise RateLimited(note)
        if "premium" in low:
            raise PremiumRequired(note)
    if "Error Message" in payload:  # unknown symbol
        return [], []
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        return [], []
    pairs: list[tuple[str, float]] = []
    for date, row in series.items():
        if not isinstance(row, dict):
            continue
        close = row.get("5. adjusted close") or row.get("4. close")
        if close is not None:
            pairs.append((date, float(close)))
    return _to_series(pairs)


def resolve_provider(preference: str, has_fmp: bool, has_alphavantage: bool) -> str:
    """Choose a provider from a preference and which API keys are present.

    ``preference`` is ``auto``/``fmp``/``alphavantage``; ``auto`` prefers FMP
    (bulk-friendly) when its key exists, else Alpha Vantage. Raises ValueError
    if the choice is impossible.
    """
    pref = (preference or "auto").lower()
    if pref == "fmp":
        if not has_fmp:
            raise ValueError("PRICE_PROVIDER=fmp but FMP_API_KEY is not set")
        return "fmp"
    if pref == "alphavantage":
        if not has_alphavantage:
            raise ValueError("PRICE_PROVIDER=alphavantage but ALPHAVANTAGE_API_KEY is not set")
        return "alphavantage"
    if pref != "auto":
        raise ValueError(f"unknown PRICE_PROVIDER {preference!r}; use auto/fmp/alphavantage")
    if has_fmp:
        return "fmp"
    if has_alphavantage:
        return "alphavantage"
    raise ValueError("no price API key set — provide FMP_API_KEY or ALPHAVANTAGE_API_KEY")
