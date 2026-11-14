# Changelog

## v1.1.0 — unreleased

The words join the numbers: plain-English insight readouts, and grounded
question-answering over the transcript corpus.

### Evaluation suite (`evals/`)
- Extraction benchmark: 30 real transcripts (30 tickers, 10 sectors,
  2014–2026, both source formats) with model-annotated (unreviewed,
  provenance-tracked) gold Q&A boundaries, a frozen 20/10 dev/holdout
  split, hash-verified reproducible retrieval from pinned dataset
  revisions, and boundary/precision/recall/contamination/abstention
  metrics (`make eval-fetch`, `make eval-extraction`)
- Scoring-correctness gate: 15 hand-calculated fixtures run in CI
  offline, documenting six quirks of the existing scoring definition
  (see evals/README.md); no scoring behavior changed
- Targeted extractor correction justified by dev-split failures: the
  analyst-handoff fallback now also recognizes "first call comes
  from/is to", "we'll go [first] to <Name> with <Firm>" (capitalized
  name required), and "now take questions from". Dev split went
  17/20 found, 14/17 within ±2 sentences → 20/20 and 20/20; holdout
  untouched (byte-identical results). Corpus-wide: abstentions
  1.94% → 1.37%; 99 boundaries move earlier (median 10% of transcript),
  recovering first Q&A exchanges previously cut off. NOTE: the
  committed feature parquet and published RAG index were built with the
  previous extractor; their next rebuild inherits these (improved)
  boundaries on ~1% of calls.

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
  current earnings season; the chunker auto-detects the two transcript
  formats (Seeking Alpha colon turns / Motley Fool name--title headers,
  98% coverage of live calls)
- Generative answer mode (`POST /ask` with `generate: true`,
  `earnings_signals/rag/generate.py`): Claude (Opus 5.5 by default,
  `ASK_LLM_MODEL` to override) composes a cited answer strictly from the
  retrieved excerpts — cite-or-refuse prompting plus a deterministic
  post-generation groundedness check (valid markers, no uncited
  sentences, lexical support per sentence); failures fall back to the
  extractive answer with the reason reported. Needs ANTHROPIC_API_KEY;
  the default extractive mode remains keyless and deterministic
- Direct-from-API ingestion: scripts/fetch_av_transcripts.py pulls
  transcripts from Alpha Vantage's EARNINGS_CALL_TRANSCRIPT endpoint
  (pre-segmented speaker turns) as a gap-filler for calls the datasets
  lack — budget-limited and resumable against the ~25 req/day free tier,
  with the attempt cache persisted via the rolling release; runs in the
  weekly refresh when ALPHAVANTAGE_API_KEY is set, skips cleanly when not
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
