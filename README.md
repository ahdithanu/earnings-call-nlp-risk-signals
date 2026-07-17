# Earnings Call NLP Risk Signals

**Detecting Hedging Language in Corporate Communications**

An NLP pipeline that measures uncertainty and hedging language in earnings call transcripts and tests whether it predicts near-term earnings. Uses the Loughran-McDonald finance-specific lexicon over the Q&A sections of S&P 500 tech earnings calls, 2013–2025.

## Why This Matters

Earnings calls contain signals beyond the numbers. Executives hedge when they're uncertain — and the unscripted Q&A exchange reveals more than prepared remarks. This project quantifies that hedging systematically and asks the question that matters: does it actually predict anything?

## Key Finding

Tested on 2,847 company-quarters across 72 large-cap tech companies:

- **Elevated hedging is a real, but modest, bearish signal.** When a company's Q&A uncertainty density rises one standard deviation above its own norm, next-quarter trailing-12M EPS growth comes in about **0.9 percentage points lower** (t = −3.12, ticker fixed effects, ticker-clustered SEs). The effect survives quarter fixed effects (−0.65pp, t = −2.42), so it is not just market-wide bad times.
- **It is weakly predictive, not merely reactive.** Density correlates slightly more with growth into the *next* quarter (r = −0.055) than with growth into the quarter being discussed (r = −0.043).
- **It is largely distinct from other sentiment — with one honest caveat.** Against all four other LM tone categories (negative, positive, litigious, constraining), the uncertainty effect holds under ticker FE (−0.58pp, p = 0.038) but softens to marginal once positive *and* negative tone are jointly absorbed under the stricter ticker+quarter FE (−0.47pp, p = 0.083). Litigious and constraining carry no signal; the shared variance is with *directional* tone — negative (bearish, ≈ −1.2 to −1.5pp per SD) and positive (bullish, ≈ +0.7 to +1.0pp per SD), each an independent predictor in its own right. So hedging is its own signal, but partly overlaps with overall optimism/pessimism.
- **It is the level that predicts, not the trend.** A company's standing hedging *level* carries the signal; the quarter-over-quarter *change* adds none (momentum coefficient marginal and, if anything, positive, p ≈ 0.05–0.06). Executives who hedge a lot matter more than executives who *started* hedging more.
- **Per-company correlations are noise.** Across 71 tickers, individual density-growth correlations center near zero and only 4 clear p < 0.05 — chance level (≈3.6 expected). An earlier version of this project reported strong per-company correlations (NVDA +0.96, MSFT −0.58) from ~3 quarters of data; on 40+ quarters per company those numbers do not replicate. Small samples tell vivid stories; panels tell true ones.

Full output: [`results/uncertainty_growth_analysis.txt`](results/uncertainty_growth_analysis.txt) and [`results/per_ticker_correlations.csv`](results/per_ticker_correlations.csv).

## Does it generalize beyond tech?

Re-running the *same* ticker-fixed-effects specification across all 11 GICS sectors of the S&P 500 — 18,836 company-quarters, 494 tickers ([`scripts/analyze_cross_sector.py`](scripts/analyze_cross_sector.py)) — the signal is **not tech-specific, but it is concentrated:**

- Strongest in **Information Technology** (−0.89pp, p = 0.003), and also significantly negative in **Financials** (−0.53pp, p = 0.003) and **Consumer Staples** (−0.43pp, p = 0.026). Negative in **8 of 11 sectors**.
- **Pooled across the entire S&P 500** (ticker + quarter fixed effects), elevated hedging still predicts weaker next-quarter EPS growth: **−0.30pp per SD, p = 0.036** — real market-wide, if about a third the size seen in tech.
- **Absent** in Health Care, Energy, and Materials (flat/positive, non-significant) — so it is a genuine but uneven effect, largest where it was first found.

Full table: [`results/cross_sector_robustness.txt`](results/cross_sector_robustness.txt) · [`results/cross_sector.csv`](results/cross_sector.csv).

## Price outcomes (post-call drift)

Does hedging predict the *stock*, not just EPS? [`scripts/fetch_prices.py`](scripts/fetch_prices.py) builds a post-call return outcome — the immediate 5-day reaction and the ~1-quarter drift — from [Financial Modeling Prep](https://financialmodelingprep.com/), and [`scripts/analyze_price_drift.py`](scripts/analyze_price_drift.py) runs the *same* fixed-effects test with the return as the outcome. The drift math is unit-tested; because the fetch needs a price-API key and open network egress, it runs via the `price-outcomes` GitHub Actions workflow — add an `FMP_API_KEY` repo secret (Settings → Secrets and variables → Actions) and dispatch it. Results land in `results/price_drift.txt`.

## Forward-Looking Signals

`scripts/latest_signals.py` turns the panel finding into a monitoring view: each ticker's most recent call is z-scored against that company's *prior* calls only (strictly out-of-sample), producing a watchlist of names whose executives are hedging unusually hard right now — [`results/latest_uncertainty_signals.csv`](results/latest_uncertainty_signals.csv). The report is only as fresh as the dataset (currently through 2025Q1); a `quarters_behind` column flags stale tickers.

**Explore it:** `web/` is a static, self-contained explorer — per-company density-vs-growth charts for all 72 tickers, a **"hedging now" watchlist** ranking who is hedging most versus their own history (out-of-sample), and the panel result up top, live at [ahdithanu.github.io/earnings-call-nlp-risk-signals](https://ahdithanu.github.io/earnings-call-nlp-risk-signals/). The validated panel ends at 2025Q1; `scripts/fetch_recent_signals.py` extends each company's density series through 2026Q2 using a live transcript source ([Rogersurf/earnings-call-transcripts](https://huggingface.co/datasets/Rogersurf/earnings-call-transcripts)), calibrated to the panel's density scale on the ~90 company-quarters the two sources share (corr 0.94). Those recent quarters carry the hedging signal only — their next-quarter EPS is not yet realized — and never enter the regression. Regenerate the page with `scripts/export_web_data.py`.

## Methodology

1. **Data:** [glopardo/sp500-earnings-transcripts](https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts) — 20,681 S&P 500 earnings-call transcripts with quarter keys, GICS sector, and trailing/forward 12-month EPS. Filtered to GICS Information Technology plus GOOGL, META, and AMZN, which GICS files under other sectors (72 tickers, 3,025 usable company-quarters; GOOG dropped as a duplicate share class of the same calls). The universe is defined in one place (`src/universe.py`); the site's headline counts are derived from the data, so adding companies is a single edit plus a rerun — no hardcoded numbers to chase.
2. **Q&A isolation:** transcripts are a single speaker-prefixed string, so the Q&A boundary is found by position-aware markers (explicit "Question-and-Answer Session" header, falling back to the operator's first-question handoff; intro announcements are ignored). 98% of transcripts split cleanly (explicit header, first-question cue, or operator analyst-handoff fallback); the rest are flagged rather than silently mis-scored.
3. **Uncertainty scoring:** negation-aware matching against the full 297-term Loughran-McDonald uncertainty category — "no material risk" does not count as risk. Density = uncertainty terms per 100 tokens, computed separately for the full call and the Q&A section.
4. **Outcome variable:** next-quarter trailing-12M EPS growth, with a calendar-gap guard so a missing quarter yields NaN instead of a fabricated "next quarter."
5. **Analysis:** per-ticker correlation sweep (Pearson + Spearman), ticker fixed-effects panel regression with ticker-clustered standard errors, growth-context test, lead-lag comparison, tone-control regressions against the other LM categories (negative, positive, litigious, constraining), and a hedging-momentum (level vs. QoQ change) check. Growth winsorized at 1%/99%; Q&A sections under 500 tokens excluded.

## Repository Structure

```
├── data/processed/
│   ├── tech_uncertainty_features.parquet   # 3,025 rows × 45 cols: ids + metrics, no transcript text
│   └── recent_uncertainty_signals.parquet  # calibrated 2025Q2–2026 density for the explorer
├── results/
│   ├── uncertainty_growth_analysis.txt     # headline analysis output
│   ├── per_ticker_correlations.csv         # all 71 per-ticker correlations
│   ├── cross_sector_robustness.txt         # finding re-run across all 11 GICS sectors
│   ├── cross_sector.csv                    # per-sector coefficients
│   ├── latest_uncertainty_signals.csv      # latest call per ticker, scored vs own history
│   └── panel_stats.json                    # derived headline stats (n, coefs, p) the site reads
├── scripts/
│   ├── inspect_dataset.py                  # Step 1: schema inspection (run before mapping fields)
│   ├── build_features.py                   # Steps 2–6: dataset → feature parquet
│   ├── fetch_recent_signals.py             # calibrated 2025Q2–2026 density (explorer only)
│   ├── analyze_uncertainty_growth.py       # panel analysis
│   ├── analyze_cross_sector.py             # S&P 500 cross-sector robustness
│   ├── fetch_prices.py                     # post-call prices from FMP (needs FMP_API_KEY)
│   └── analyze_price_drift.py              # does hedging predict returns, not just EPS?
│   ├── latest_signals.py                   # forward-looking monitoring report
│   └── export_web_data.py                  # renders self-contained web/index.html from the parquet
├── web/                                    # self-contained explorer: index.template.html (source) → index.html (generated, data inlined)
├── src/
│   ├── lexicon.py                          # LM lexicon loader (refuses truncated lists)
│   ├── uncertainty.py                      # tokenizer + negation-aware uncertainty counting
│   ├── qa_extract.py                       # Q&A boundary detection
│   ├── features.py                         # forward EPS shift with quarter-gap guard
│   ├── price_drift.py                      # post-call return math (unit-tested)
│   └── universe.py                         # the universe rule — single edit point to add companies
├── tests/                                  # 32 unit tests
├── lm_uncertainty_terms.txt                # full 297-term LM uncertainty category
├── lm_negative_terms.txt                   # full 2,355-term LM negative category (tone control)
├── lm_positive_terms.txt                   # full 354-term LM positive category (tone control)
├── lm_litigious_terms.txt                  # full 904-term LM litigious category (tone control)
└── lm_constraining_terms.txt               # full 184-term LM constraining category (tone control)
```

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/            # 32 tests
python -m scripts.build_features   # rebuilds the parquet (downloads dataset on first run)
python -m scripts.analyze_uncertainty_growth
python -m scripts.latest_signals       # score each ticker's most recent call
python -m scripts.fetch_recent_signals # calibrated 2025Q2–2026 quarters (explorer)
python -m scripts.export_web_data      # render web/index.html (panel + recent, data inlined)
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

- **Transcripts + EPS (validated panel, 2013–2025Q1):** [glopardo/sp500-earnings-transcripts](https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts) (Hugging Face)
- **Recent transcripts (explorer signal, through 2026):** [Rogersurf/earnings-call-transcripts](https://huggingface.co/datasets/Rogersurf/earnings-call-transcripts) (Hugging Face)
- **Uncertainty Dictionary:** [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

## Roadmap

- [x] Q&A isolation recovery — operator analyst-handoff fallback (95.8% → 98.4%)
- [x] LM tone controls — negative, litigious, constraining (uncertainty survives all three)
- [x] Price-based outcomes scaffolded — post-call drift fetch + regression + CI workflow (activate with an `FMP_API_KEY` secret)

## License

MIT License

## Author

**Ahdithan Uthayakumar**
[LinkedIn](https://linkedin.com/in/ahdithan) · [GitHub](https://github.com/ahdithanu)
