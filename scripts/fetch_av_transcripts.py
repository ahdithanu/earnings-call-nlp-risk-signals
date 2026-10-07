"""Pull earnings-call transcripts from Alpha Vantage (EARNINGS_CALL_TRANSCRIPT).

The direct-from-API ingestion path: where the two Hugging Face datasets
update on their maintainers' cadence, this endpoint serves a transcript
days after the call. It feeds the SAME artifact — scripts/build_rag_index.py
picks up the cache written here as a third chunk source (rag/av.py) that
fills (ticker, quarter) pairs the dataset sources don't cover yet.

Built around Alpha Vantage's rate limits (free tier ~25 requests/day):

  - every attempt is recorded in an append-only cache
    (data/av_transcripts/av_transcripts.jsonl.gz, latest record per pair
    wins); successful fetches are never re-requested, failed/empty ones
    retry after RETRY_DAYS, so coverage ACCUMULATES across weekly runs —
    the same resumable pattern as the price fetcher
  - at most AV_TRANSCRIPT_BUDGET requests per run (default 25), spaced
    AV_SECONDS_BETWEEN apart (default 12s ≈ the free tier's 5/min);
    a rate-limit notice stops the run immediately, keeping what it has
  - never-attempted pairs are fetched before stale retries, so the tail
    of the alphabet is not starved

The workflow seeds the cache from the rolling `rag-index` release and
re-uploads it after the run, so state survives between ephemeral runners.

Quarter semantics caveat: the API's ``quarter`` parameter is the FISCAL
quarter. For most of the S&P 500 fiscal == calendar, but offset-fiscal-year
companies (AAPL, MSFT, ...) will have their call labeled by fiscal quarter
here. The index build therefore uses these transcripts only to fill pairs
the calendar-labeled dataset sources lack.

Usage:  ALPHAVANTAGE_API_KEY=... python -m scripts.fetch_av_transcripts
        [--budget N] [--cache data/av_transcripts/av_transcripts.jsonl.gz]
"""

import argparse
import gzip
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from earnings_signals.price_providers import ALPHAVANTAGE_URL
from earnings_signals.rag.av import parse_av_response

PANEL = "data/processed/sp500_uncertainty_features.parquet"
CACHE = "data/av_transcripts/av_transcripts.jsonl.gz"
RETRY_DAYS = 7
N_QUARTER_LABELS = 2  # current + previous calendar quarter


def quarter_labels(today: datetime, n: int = N_QUARTER_LABELS) -> list[str]:
    """The n most recent calendar-quarter labels, newest first."""
    year, q = today.year, (today.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        out.append(f"{year}Q{q}")
        q -= 1
        if q == 0:
            year, q = year - 1, 4
    return out


def load_cache(path: Path) -> dict[tuple[str, str], dict]:
    """Latest cache record per (symbol, quarter)."""
    records: dict[tuple[str, str], dict] = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[(rec["symbol"], rec["quarter"])] = rec
    return records


def select_targets(
    tickers: list[str], labels: list[str], cache: dict[tuple[str, str], dict], now: datetime
) -> list[tuple[str, str]]:
    """Pairs worth requesting: never-attempted first, then stale retries."""
    retry_before = (now - timedelta(days=RETRY_DAYS)).strftime("%Y-%m-%d")
    fresh: list[tuple[str, str]] = []
    stale: list[tuple[str, str]] = []
    for label in labels:  # newest quarter first — freshness is the point
        for t in tickers:
            rec = cache.get((t, label))
            if rec is None:
                fresh.append((t, label))
            elif rec["status"] != "ok" and rec["fetched"] < retry_before:
                stale.append((t, label))
    return fresh + stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget", type=int, default=int(os.environ.get("AV_TRANSCRIPT_BUDGET", 25))
    )
    parser.add_argument(
        "--sleep", type=float, default=float(os.environ.get("AV_SECONDS_BETWEEN", 12))
    )
    parser.add_argument("--cache", default=CACHE)
    args = parser.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("ALPHAVANTAGE_API_KEY is not set — nothing fetched (not an error)")
        return 0

    tickers = sorted(pd.read_parquet(PANEL, columns=["ticker"])["ticker"].unique())
    now = datetime.now(UTC)
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    targets = select_targets(list(tickers), quarter_labels(now), cache, now)
    print(f"cache: {len(cache)} records; eligible targets: {len(targets)}; budget: {args.budget}")

    new_records: list[dict] = []
    counts = {"ok": 0, "empty": 0, "error": 0}
    for i, (symbol, label) in enumerate(targets[: args.budget]):
        if i:
            time.sleep(args.sleep)
        try:
            resp = requests.get(
                ALPHAVANTAGE_URL,
                params={
                    "function": "EARNINGS_CALL_TRANSCRIPT",
                    "symbol": symbol,
                    "quarter": label,
                    "apikey": api_key,
                },
                timeout=60,
            )
            result = parse_av_response(resp.json())
        except (requests.RequestException, ValueError) as exc:
            result = parse_av_response({"Error Message": f"request failed: {exc}"})
        if result.status == "rate_limited":
            print(f"rate limited after {i} requests ({result.detail}) — stopping, cache kept")
            break
        rec = {
            "symbol": symbol,
            "quarter": label,
            "fetched": now.strftime("%Y-%m-%d"),
            "status": result.status,
        }
        if result.status == "ok":
            rec["turns"] = result.turns
        else:
            rec["detail"] = result.detail
        new_records.append(rec)
        counts[result.status] += 1
        print(
            f"  {symbol} {label}: {result.status}"
            + (f" ({len(result.turns)} turns)" if result.turns else "")
        )

    if new_records:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "at", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    merged = load_cache(cache_path)
    print(
        f"this run: {counts['ok']} transcripts, {counts['empty']} empty, {counts['error']} errors; "
        f"cache now {sum(1 for r in merged.values() if r['status'] == 'ok')} transcripts "
        f"/ {len(merged)} attempted pairs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
