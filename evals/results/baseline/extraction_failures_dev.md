# Extraction benchmark failures — split: dev
generated: 2026-11-14T05:09:05+00:00
annotations: model-annotated: labeled by Claude in this repository's development session (2026-11-14) by READING each source transcript around candidate transitions found with a cue set broader than the extractor's (evals/show_candidates.py). NOT derived from extractor output, and NOT human-reviewed. (review status: see evals/annotations.json)

evaluated: 20 | invalid (excluded): 0
outcomes: {'false_abstain': 3, 'found_vs_gold': 17}

- **glopardo-AMT-2024Q4** [abstain] false_abstain (gold_start=26570, predicted=None)
- **glopardo-UAL-2016Q3** [abstain] false_abstain (gold_start=15756, predicted=None)
- **glopardo-UNH-2025Q1** [abstain] false_abstain (gold_start=21657, predicted=None)
- **glopardo-AOS-2023Q3** [handoff] found_vs_gold: sentence_delta=94, char_delta=8769, contamination=0.000, recall=0.647
- **glopardo-APH-2024Q1** [handoff] found_vs_gold: sentence_delta=32, char_delta=2993, contamination=0.000, recall=0.901
- **glopardo-CRL-2018Q1** [handoff] found_vs_gold: sentence_delta=18, char_delta=2365, contamination=0.000, recall=0.920
