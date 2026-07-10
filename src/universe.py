"""The analysis universe — the ONE place that decides which companies are in.

To change the universe, edit ``UNIVERSE_SECTORS`` and/or ``EXTRA_TICKERS``
here (or ``select_universe`` for a fully custom rule) and rerun the pipeline.
Everything downstream — scoring, the panel, the explorer, the counts shown on
the site — follows from this; no company is wired in anywhere else. The site's
headline numbers (n, ticker count) are derived from the data, not hardcoded,
so a universe change needs no edits outside this file.
"""

import pandas as pd

# GICS sectors included wholesale. The dataset's own sector field is the
# preferred selector; add sectors here to widen coverage (e.g. the whole
# S&P 500 is every sector in glopardo).
UNIVERSE_SECTORS = frozenset({"Information Technology"})

# Individual tickers to include regardless of sector — mega-caps GICS files
# elsewhere (GOOGL/META in Communication Services, AMZN in Consumer
# Discretionary). GOOG is intentionally absent: the dataset carries Alphabet's
# calls under both share classes, and both would double-count the same calls.
EXTRA_TICKERS = frozenset({"GOOGL", "META", "AMZN"})


def select_universe(df: pd.DataFrame) -> "pd.Series":
    """Boolean mask selecting the analysis universe from the raw dataset.

    A row is in the universe if its GICS sector is in ``UNIVERSE_SECTORS`` or
    its ticker is in ``EXTRA_TICKERS``.
    """
    return df["sector"].isin(UNIVERSE_SECTORS) | df["ticker"].isin(EXTRA_TICKERS)


# Hardcoded large-cap tech list — a FALLBACK for a source with no sector
# field. Unused while the dataset provides GICS sectors; kept for that case.
TECH_TICKERS = frozenset(
    {
        "NVDA", "MSFT", "AAPL", "GOOGL", "META", "AMZN", "AMD", "INTC",
        "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "AVGO", "TXN", "NOW",
        "IBM", "MU",
    }
)
