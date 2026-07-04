"""Tech-company universe for Step 2 filtering.

Preferred path is the dataset's own sector field; this hardcoded list is the
FALLBACK when no sector field exists in the source data.
"""

# Large-cap tech fallback list (to be refined later).
TECH_TICKERS = frozenset(
    {
        "NVDA", "MSFT", "AAPL", "GOOGL", "META", "AMZN", "AMD", "INTC",
        "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "AVGO", "TXN", "NOW",
        "IBM", "MU",
    }
)

# Companies the project treats as tech even though GICS files them under
# Communication Services (GOOGL, META) or Consumer Discretionary (AMZN).
# Unioned with the sector filter in scripts/build_features.py. GOOG is
# deliberately absent: the dataset carries Alphabet's calls under both share
# classes, and including both would double-count the same transcripts.
EXTRA_TECH_TICKERS = frozenset({"GOOGL", "META", "AMZN"})
