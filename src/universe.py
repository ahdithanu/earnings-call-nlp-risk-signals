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
