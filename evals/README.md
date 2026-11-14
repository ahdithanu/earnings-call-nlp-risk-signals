# Evaluation Suite

Measures two things the statistical robustness checks in `results/` can
NOT establish: **extraction accuracy** (does `extract_qa` cut transcripts
where the Q&A actually starts?) and **scoring correctness** (does
`count_uncertainty` compute what its specification says?). Panel
regressions survive noise; they don't tell you the measurement is right.

## One-command runs

```bash
make eval            # scoring-correctness fixtures (offline, CI gate)
make eval-fetch      # download + hash-verify the 30 benchmark transcripts
make eval-extraction # extraction benchmark on the dev split -> evals/results/
```

`python -m evals.run_extraction_eval --split dev|holdout|all` is the
underlying command; it writes machine-readable JSON plus a readable
failure report under `evals/results/`.

## The benchmark (30 real transcripts — an initial set with limited coverage)

30 real transcripts, 30 unique tickers, 10 GICS sectors, 2014–2026, both
source formats (Seeking Alpha-style panel source; Motley Fool-style live
source). Selected for DIFFICULTY coverage — operator-announcement traps,
repeated Q&A mentions, missing headers, analyst handoffs, "first call"
phrasing, a 4%-position boundary, and genuinely missing Q&A — using the
extractor's own structural signals (`evals/build_manifest.py`). Two
consequences, stated plainly:

- selection is biased toward hard cases, so aggregate rates here do NOT
  estimate corpus-wide accuracy;
- 30 transcripts is an initial benchmark. It covers each difficulty
  class with a handful of examples, not enough for tight per-class rates.

Transcripts are **not committed** (they are redistributed by their public
HF source datasets, not by this repo; dataset revisions are pinned in
`evals/manifest.json`). `python -m evals.fetch_benchmark` retrieves them
reproducibly and verifies each sha256. Only short anchor excerpts
(≤100 chars) appear in the annotations.

## Annotations

`evals/annotations.json`, produced per `evals/ANNOTATION_GUIDE.md` by
reading each transcript. **Provenance: model-annotated (this repository's
Claude development session), unreviewed.** They are provisional labels,
tracked with explicit review status; one case (`glopardo-ERIE-2017Q4`) is
flagged ambiguous. Until human review, every extraction number below the
correctness gates is **informational, not an accuracy claim**.

Annotations were frozen (committed) before any extraction change. The
20/10 dev/holdout split was assigned at selection time, before
annotation, stratified by difficulty bucket; tickers are unique across
the benchmark, so there are no identical or near-duplicate transcripts to
keep together. Holdout results are reported but never used to tune; if a
holdout failure ever motivates a change, that transcript is disclosed as
burnt and replaced.

## Extraction metrics (defined before the first run)

Per transcript: outcome class (`found_vs_gold` / `false_abstain` /
`false_find` / `correct_abstain` / `invalid`), signed boundary error in
characters and in sentences, exact-boundary flag (same sentence),
tolerance flag (|Δ| ≤ 2 sentences — a documented allowance for
legitimately different utterance-start conventions), token precision and
recall over the gold Q&A span, and prepared-remarks contamination
(share of extracted tokens that precede the gold boundary). `invalid`
(unfetched/hash-mismatched/anchor-mismatched transcripts) is reported and
excluded from denominators — a technical failure is never scored as an
extraction result. Raw counts and per-transcript rows accompany every
aggregate. Full definitions: `evals/metrics.py` docstring.

### Proposed acceptance thresholds (recorded before the baseline run)

These become enforceable gates only once annotations are human-reviewed;
until then the runner reports pass/fail informationally and always exits 0.

| metric (dev split) | threshold |
|---|---|
| boundary within ±2 sentences, among gold-positive found | ≥ 80% |
| false finds on gold-absent transcripts | 0 |
| mean contamination (prepared-remarks tokens in extraction) | ≤ 2% |
| mean recall of gold Q&A span | ≥ 90% |

Thresholds must not be weakened to hide a failure; a threshold change
requires a written rationale here.

## Scoring correctness (CI gate, offline)

`evals/scoring_fixtures.json` + `tests/test_scoring_fixtures.py`: 15
hand-calculated passages checked against the implementation on every CI
run. This is a **correctness** gate (implementation == specification),
deterministic and network-free.

### The scoring definition under test

- tokenizer: `[a-z]+(?:'[a-z]+)?` over lowercased text — lowercase ASCII
  letter runs, internal apostrophes kept;
- match: exact token membership in the committed 297-term LM uncertainty
  list (inflections only as listed; no stemming);
- negation: a match is excluded when any of the 3 preceding tokens is in
  {no, not, never, without, don't, doesn't, didn't, wasn't, isn't,
  aren't}; exclusions are tallied separately;
- density: `uncertainty_count / total_tokens * 100`; `None` when there
  are no tokens.

### Known inconsistencies the fixtures expose (documented, not "fixed")

1. **Numerals are not tokens**: "2026" and "15" are absent from
   `total_tokens`, so density on number-heavy text is computed over words
   only (`numbers_not_in_denominator`).
2. **Possessives never match**: "risk's" is one token and no lexicon
   entry, so possessive mentions are silently missed
   (`hyphen_split_and_possessive_miss`).
3. **Hyphenated compounds split**: "risk-adjusted" contributes a counted
   "risk" (same fixture).
4. **Negator overreach**: one negator suppresses every lexicon term in
   its 3-token reach, not only the term it governs
   (`contraction_negators_and_overreach`).
5. **Missing negators**: cannot / can't / won't / couldn't / wouldn't are
   not in the negator list, so "we cannot rule out risk" counts
   (`missing_negators_cannot_family`).
6. **ASCII-only tokenizer**: accented words split into fragments and
   inflate the denominator (`ascii_only_tokenizer`).

These are properties of the existing specification. Changing any of them
is a scoring-definition change that would shift every density in the
panel and must be versioned, not slipped into a fix.

## Contextual validity is a separate, unmeasured question

Matching a lexicon term is **not** evidence that a passage expresses
uncertainty ("may" in "May I say" is a counted false positive by
construction; hedging without lexicon words is an uncounted false
negative). Nothing in this suite measures that, and no semantic
precision/recall is claimed anywhere in this repository.

Future protocol, for when human annotation is available:
1. sample N≈200 lexicon-hit sentences (stratified by term frequency) and
   N≈200 no-hit sentences from executive Q&A text;
2. two annotators label each sentence: does the speaker express
   uncertainty/hedging about the business? (guide with examples;
   adjudicate disagreements; report inter-annotator agreement);
3. report semantic precision (hit sentences judged uncertain) and a
   false-negative rate (no-hit sentences judged uncertain) with CIs;
4. only then attach semantic claims to the density signal.

## Limitations (read before quoting numbers)

- Annotations are model-produced and unreviewed; the annotator read the
  text but is the same model family that develops this repo.
- 30 transcripts; per-bucket denominators are 1–6.
- Difficulty-enriched selection: aggregates are not corpus rates.
- Boundary conventions involve judgment at utterance granularity; the
  ±2-sentence tolerance absorbs most but not all of it.
- The extraction benchmark needs network once (`make eval-fetch`);
  routine CI runs only the offline correctness gates.
