"""The analysis universe — the ONE place that decides which companies are in.

To change the universe, edit ``UNIVERSE_SECTORS``, ``EXTRA_TICKERS`` and/or
``EXCLUDE_TICKERS`` here (or ``select_universe`` for a fully custom rule)
and rerun the pipeline. Everything downstream — scoring, the panel, the
explorer, the counts shown on the site — follows from this; no company is
wired in anywhere else. The site's headline numbers (n, ticker count) are
derived from the data, not hardcoded, so a universe change needs no edits
outside this file.

The primary panel is the FULL S&P 500 (every GICS sector in the dataset);
per-sector heterogeneity is an analysis dimension, not a filter. The
original tech-only framing lives on as the sector breakdown in
scripts/analyze_cross_sector.py.
"""

import pandas as pd

# GICS sectors included wholesale — all 11, i.e. the whole S&P 500.
UNIVERSE_SECTORS = frozenset({
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
})

# Individual tickers to include regardless of sector. Empty now that every
# sector is in; kept for narrower future universes.
EXTRA_TICKERS: frozenset[str] = frozenset()

# Dual-share-class duplicates: the dataset carries the same calls under both
# tickers (verified identical CIK), which would double-count them. Keep
# GOOGL/NWSA, drop the twins.
EXCLUDE_TICKERS = frozenset({"GOOG", "NWS"})


def select_universe(df: pd.DataFrame) -> "pd.Series":
    """Boolean mask selecting the analysis universe from the raw dataset.

    A row is in the universe if its GICS sector is in ``UNIVERSE_SECTORS``
    or its ticker is in ``EXTRA_TICKERS``, and its ticker is not in
    ``EXCLUDE_TICKERS``.
    """
    included = df["sector"].isin(UNIVERSE_SECTORS) | df["ticker"].isin(EXTRA_TICKERS)
    return included & ~df["ticker"].isin(EXCLUDE_TICKERS)


# Hardcoded large-cap tech list — a FALLBACK for a source with no sector
# field. Unused while the dataset provides GICS sectors; kept for that case.
TECH_TICKERS = frozenset(
    {
        "NVDA", "MSFT", "AAPL", "GOOGL", "META", "AMZN", "AMD", "INTC",
        "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "AVGO", "TXN", "NOW",
        "IBM", "MU",
    }
)
