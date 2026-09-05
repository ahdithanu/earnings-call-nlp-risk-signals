# Design notes

The decisions that shaped this project, and why. The README carries the
findings; this documents the engineering and methodological trade-offs a
reviewer would ask about.

## Data

**Trailing-12M EPS as the outcome.** The dataset carries no single-quarter
EPS; `eps12mtrailing_eoq` is the only realized-earnings series (98.6%
coverage). QoQ growth of a TTM series has 75% mechanical window overlap, so
`earnings_signals/robustness_single_quarter.py` re-estimates the headline on the
non-overlapping YoY innovation — same conclusion, which is why the smoother
series is acceptable as the primary outcome.

**Dual share classes are deduplicated by CIK.** GOOG/GOOGL and NWS/NWSA carry
the same calls twice; one twin of each is excluded in
`earnings_signals/universe.py`, the single place the universe is defined.

**Two-layer freshness.** The panel (EPS-bearing, frozen dataset) is rebuilt
manually; a weekly cron advances only the density-only "recent signals"
layer from a live transcript source, calibrated to the panel's scale on the
overlap sample. Recent quarters never enter regressions — their outcomes
don't exist yet.

## Text processing

**Negation-aware counting.** A lexicon match is suppressed when a negator
occurs within the 3 preceding tokens ("no material risk"), and suppressed
counts are reported (`negation_excluded_*`), not hidden. One tokenizer and
one counting rule serve every lexicon category and every scope, so
densities are comparable across all of them.

**Isolation is layered, and every layer reports failure.** Q&A boundary →
executive-only turns → CEO/CFO role split. Each layer has a flag
(`qa_isolated`, `qa_exec_isolated`, `role_attributed` + `role_source`) and
missing coverage yields nulls, never substituted broader text. The
transcript corpus changes format in 2019 (roster header disappears); roles
are parsed from the roster before and from IR intro prose after, and the
two independent parsers agreeing at 87.7% surname continuity across the
boundary (most disagreements being genuine executive transitions) is the
validation that both are right.

**Why executive-only scoring exists.** Analyst questions are themselves
uncertainty-heavy ("what risks do you see"), inflating full-Q&A density
levels ~25%. Within-ticker z-scoring absorbs most of that in the panel, but
the exec-only scope removes it at the source — and at market scale it
carries slightly more signal than the full Q&A.

## Statistics

**Within-company design.** Density levels differ across companies and call
styles, so the predictor is z-scored within ticker and every spec carries
ticker fixed effects with ticker-clustered SEs; the stricter spec adds
calendar-quarter FE. Growth is winsorized 1%/99% (TTM EPS near zero makes
percentage growth explode).

**Caveats are findings.** The tone-control result (uncertainty's
incremental effect weakens under quarter FE once directional tone is
absorbed) and the per-ticker noise result are reported as prominently as
the headline. An early version of the project claimed NVDA r = +0.96 from
3 quarters; the panel retracts it, and the retraction stays in the README.

## Engineering

**One source of truth per fact.** The universe (`universe.py`), thresholds
(module constants), headline stats (`results/panel_stats.json`, derived,
read by the site), dependency versions (`pyproject.toml`). README numbers
are copied from `results/`, never invented.

**Fail loudly, especially unattended.** Lexicon loaders refuse truncated
files; the data-quality gate (`scripts/validate_features.py`) checks
structure, integrity, and plausibility after every build and in CI; the
weekly cron files a GitHub issue on failure instead of rotting silently.

**The service is stateless.** The API loads lexicons once and scores
caller-supplied text; the only file it reads is the committed watchlist.
That keeps the container free-tier-sized and the demo durable.
