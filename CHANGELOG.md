# Changelog

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
