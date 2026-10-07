# Changelog

## v1.1.0 — unreleased

The words join the numbers: plain-English insight readouts, and grounded
question-answering over the transcript corpus.

### Ask the transcripts (`POST /ask`, `earnings_signals/rag/`)
- Speaker-turn chunking of executive Q&A answers (55,289 chunks, most
  recent 4 quarters) with ticker/sector/quarter/speaker metadata
- Hybrid retrieval: BM25 + static dense embeddings (model2vec
  `potion-base-8M` — numpy inference, no torch), fused with reciprocal
  rank fusion; ticker filters masked before ranking (entitlement model)
- Structurally grounded answers: verbatim corpus sentences only, each
  cited; out-of-KB questions refused on a threshold calibrated against
  measured junk-vs-real score distributions
- Retrieval eval harness (`make rag-eval` → `results/rag_eval.txt`):
  recall@K + MRR ablation across bm25/dense/hybrid — hybrid reaches 100%
  call-level recall@10 — plus a refusal check (25/25 junk refused)
- Live ingestion: the index merges the validated panel source (up to its
  watermark) with the Rogersurf live source beyond it, so /ask covers the
  current earnings season (through 2026Q2); the chunker auto-detects the
  two transcript formats (Seeking Alpha colon turns / Motley Fool
  name--title headers, 98% coverage of live calls)
- The index is a weekly build artifact, not a committed file: the Monday
  refresh workflow rebuilds and publishes it to the rolling `rag-index`
  GitHub release; `make rag-fetch` downloads it, and Docker/CI builds
  bake it (with the embedding model) into the image — no network at
  runtime, no 36MB/week of git history

### Insight readouts (`GET /insight/{ticker}`, `earnings_signals/insights.py`)
- Deterministic plain-English readouts for every company's latest call:
  level vs own history, CEO-vs-CFO driver, hedging-vs-bad-news character,
  driving terms vs the company's usual vocabulary, excerpt receipts
- Powers the explorer's per-company cards and `results/weekly_brief.md`

### Engineering
- Price outcomes: pluggable price provider — Alpha Vantage in addition to
  FMP (`earnings_signals/price_providers.py`). `PRICE_PROVIDER` selects
  auto/fmp/alphavantage; the fetch is resumable so Alpha Vantage's ~25/day
  free tier accumulates coverage across runs. Response parsing is
  unit-tested. The price-outcomes workflow now takes either secret.
- 71 unit tests (hermetic RAG tests via a hashing embedder — no model
  download in CI)

## v1.0.0 — 2026-09-03

First tagged release: the project graduates from analysis scripts to a
packaged, gated, deployable system.

### Findings (full S&P 500 panel, 19,081 company-quarters, 494 tickers)
- +1 SD in Q&A uncertainty density → −0.55pp next-quarter TTM EPS growth
  (t = −3.87, ticker FE); −0.29pp (p = 0.038) with quarter FE
- The signal concentrates in executive speech, specifically the CEO's
  (−0.57pp, p = 0.001; CFO n.s.), and holds across both transcript-format eras
- Sector-uneven: strongest in Information Technology, Financials, Consumer
  Staples; absent in Health Care, Energy, Materials
- Honest caveats retained: overlaps with directional tone under the
  strictest spec; level predicts, trend doesn't

### Engineering
- Installable package (`earnings_signals`), pyproject-managed deps
- FastAPI scoring service (`/score`, `/signals`, `/healthz`), Dockerized,
  published to GHCR on version tags
- CI gates: ruff lint + format, mypy, pytest (3.11/3.12) with coverage,
  Docker build + container smoke test, data-quality gate
- Data-quality validation after every build; weekly refresh cron files a
  GitHub issue on failure
- Explorer: 494-company static site with optional live-scoring section
- 49 unit tests
