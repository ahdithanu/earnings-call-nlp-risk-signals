# Earnings Call NLP Risk Signals

**Detecting Hedging Language in Corporate Communications**

An NLP pipeline that measures uncertainty and hedging language in earnings call transcripts and tests whether it predicts near-term earnings. Uses the Loughran-McDonald finance-specific lexicon over the Q&A sections of S&P 500 tech earnings calls, 2013–2025.

## Why This Matters

Earnings calls contain signals beyond the numbers. Executives hedge when they're uncertain — and the unscripted Q&A exchange reveals more than prepared remarks. This project quantifies that hedging systematically and asks the question that matters: does it actually predict anything?

## Key Finding

Tested on 2,657 company-quarters across 69 large-cap tech companies:

- **Elevated hedging is a real, but modest, bearish signal.** When a company's Q&A uncertainty density rises one standard deviation above its own norm, next-quarter trailing-12M EPS growth comes in about **0.9 percentage points lower** (t = −3.10, ticker fixed effects, ticker-clustered SEs). The effect survives quarter fixed effects (−0.65pp, t = −2.45), so it is not just market-wide bad times.
- **It is weakly predictive, not merely reactive.** Density correlates slightly more with growth into the *next* quarter (r = −0.059) than with growth into the quarter being discussed (r = −0.049).
- **Per-company correlations are noise.** Across 68 tickers, individual density-growth correlations center near zero and only 3 clear p < 0.05 — chance level. An earlier version of this project reported strong per-company correlations (NVDA +0.96, MSFT −0.58) from ~3 quarters of data; on 40+ quarters per company those numbers do not replicate. Small samples tell vivid stories; panels tell true ones.

Full output: [`results/uncertainty_growth_analysis.txt`](results/uncertainty_growth_analysis.txt) and [`results/per_ticker_correlations.csv`](results/per_ticker_correlations.csv).

## Methodology

1. **Data:** [glopardo/sp500-earnings-transcripts](https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts) — 20,681 S&P 500 earnings-call transcripts with quarter keys, GICS sector, and trailing/forward 12-month EPS. Filtered to GICS Information Technology (69 tickers, 2,895 usable company-quarters).
2. **Q&A isolation:** transcripts are a single speaker-prefixed string, so the Q&A boundary is found by position-aware markers (explicit "Question-and-Answer Session" header, falling back to the operator's first-question handoff; intro announcements are ignored). 96% of transcripts split cleanly; the rest are flagged rather than silently mis-scored.
3. **Uncertainty scoring:** negation-aware matching against the full 297-term Loughran-McDonald uncertainty category — "no material risk" does not count as risk. Density = uncertainty terms per 100 tokens, computed separately for the full call and the Q&A section.
4. **Outcome variable:** next-quarter trailing-12M EPS growth, with a calendar-gap guard so a missing quarter yields NaN instead of a fabricated "next quarter."
5. **Analysis:** per-ticker correlation sweep (Pearson + Spearman), ticker fixed-effects panel regression with ticker-clustered standard errors, growth-context test, and lead-lag comparison. Growth winsorized at 1%/99%; Q&A sections under 500 tokens excluded.

## Repository Structure

```
├── data/processed/
│   └── tech_uncertainty_features.parquet   # 2,895 rows × 29 cols: ids + metrics, no transcript text
├── results/
│   ├── uncertainty_growth_analysis.txt     # headline analysis output
│   └── per_ticker_correlations.csv         # all 68 per-ticker correlations
├── scripts/
│   ├── inspect_dataset.py                  # Step 1: schema inspection (run before mapping fields)
│   ├── build_features.py                   # Steps 2–6: dataset → feature parquet
│   └── analyze_uncertainty_growth.py       # panel analysis
├── src/
│   ├── lexicon.py                          # LM lexicon loader (refuses truncated lists)
│   ├── uncertainty.py                      # tokenizer + negation-aware uncertainty counting
│   ├── qa_extract.py                       # Q&A boundary detection
│   ├── features.py                         # forward EPS shift with quarter-gap guard
│   └── universe.py                         # hardcoded tech-ticker fallback (unused; dataset has a sector field)
├── tests/                                  # 19 unit tests
└── lm_uncertainty_terms.txt                # full 297-term LM uncertainty category
```

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/            # 19 tests
python -m scripts.build_features   # rebuilds the parquet (downloads dataset on first run)
python -m scripts.analyze_uncertainty_growth
```

Score any text directly:

```python
from src.lexicon import load_uncertainty_terms
from src.uncertainty import count_uncertainty

lexicon = load_uncertainty_terms()
result = count_uncertainty(
    "We believe revenue could potentially exceed expectations, "
    "though there is no material risk to our guidance.", lexicon,
)
print(result.uncertainty_count, result.negation_excluded, result.density)
```

## Data Sources

- **Transcripts + EPS:** [glopardo/sp500-earnings-transcripts](https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts) (Hugging Face)
- **Uncertainty Dictionary:** [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

## Roadmap

- [ ] Expand beyond GICS Information Technology (GOOGL/META/AMZN sit in other sectors)
- [ ] Recover the ~4% of transcripts where Q&A isolation fails
- [ ] Additional L-M categories (litigious, negative, constraining) as parallel features
- [ ] Price-based outcomes (post-call drift) alongside EPS growth

## License

MIT License

## Author

**Ahdithan Uthayakumar**
[LinkedIn](https://linkedin.com/in/ahdithan) · [GitHub](https://github.com/ahdithanu)
