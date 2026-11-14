# Extraction benchmark failures — split: holdout
generated: 2026-11-14T05:09:44+00:00
annotations: model-annotated: labeled by Claude in this repository's development session (2026-11-14) by READING each source transcript around candidate transitions found with a cue set broader than the extractor's (evals/show_candidates.py). NOT derived from extractor output, and NOT human-reviewed. (review status: see evals/annotations.json)

evaluated: 10 | invalid (excluded): 0
outcomes: {'correct_abstain': 1, 'false_abstain': 1, 'found_vs_gold': 8}

- **glopardo-BEN-2019Q3** [handoff] found_vs_gold: sentence_delta=56, char_delta=5755, contamination=0.000, recall=0.880
- **rogersurf-FOXA-2026Q2** [rs_abstain] false_abstain (gold_start=10451, predicted=None)
