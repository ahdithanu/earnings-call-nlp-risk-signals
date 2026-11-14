"""Extraction-quality metrics against gold Q&A-boundary annotations.

Definitions (fixed before any tuning; see evals/README.md for rationale):

Outcome classes per transcript (mutually exclusive):
  found_vs_gold   gold has a Q&A boundary and the extractor returned one
  false_abstain   gold has a boundary, extractor returned None
  false_find      gold says no identifiable Q&A, extractor returned one
  correct_abstain both sides say none
  invalid         transcript could not be loaded/verified (technical
                  failure — reported, excluded from accuracy denominators)

Boundary metrics (over found_vs_gold only):
  char_delta          predicted_offset - gold_offset (signed; negative =
                      extractor cut early, positive = cut late)
  sentence_delta      signed count of sentence endings strictly between
                      the two offsets (0 = same sentence). Sentence
                      endings are occurrences of [.!?] followed by
                      whitespace/quote — a documented approximation.
  exact               sentence_delta == 0
  within_tolerance    |sentence_delta| <= TOLERANCE_SENTENCES (= 2):
                      boundary conventions legitimately differ by an
                      operator sentence or a section-header line, and a
                      <=2-sentence miss changes Q&A token counts by <0.1%
                      on a typical 8k-token section.

Content metrics (token = whitespace-split word; over found_vs_gold):
  gold span       [gold_start, gold_end) — gold_end defaults to the end
                  of the transcript; annotated earlier when trailing
                  non-call content (participant lists, scraper footers)
                  is present.
  extracted span  [predicted, end of transcript) — the extractor's
                  actual contract (extract_qa returns the suffix).
  precision       tokens of extracted that lie inside gold span / tokens extracted
  recall          tokens of extracted inside gold span / tokens of gold span
  contamination   tokens of extracted BEFORE gold_start (prepared
                  remarks leakage) / tokens extracted. Precision < 1 with
                  contamination 0 means tail junk after gold_end, not
                  prepared-remarks leakage.
"""

import re
from dataclasses import dataclass, field

from earnings_signals.qa_extract import find_qa_start

TOLERANCE_SENTENCES = 2

_SENT_END_RE = re.compile(r"[.!?]+[\s\"')\]]")


def sentence_delta(text: str, pred: int, gold: int) -> int:
    """Signed sentence distance: + means pred is later than gold."""
    lo, hi = min(pred, gold), max(pred, gold)
    n = len(_SENT_END_RE.findall(text[lo:hi]))
    return n if pred >= gold else -n


def _tokens_in(text: str, start: int, end: int) -> int:
    return len(text[start:end].split())


@dataclass
class TranscriptResult:
    id: str
    split: str
    bucket: str
    outcome: str  # found_vs_gold | false_abstain | false_find | correct_abstain | invalid
    gold_start: int | None = None
    predicted: int | None = None
    char_delta: int | None = None
    sentence_delta: int | None = None
    exact: bool | None = None
    within_tolerance: bool | None = None
    precision: float | None = None
    recall: float | None = None
    contamination: float | None = None
    ambiguous: bool = False
    note: str = ""


def evaluate_transcript(
    text: str,
    annotation: dict,
    *,
    id: str,
    split: str,
    bucket: str,
) -> TranscriptResult:
    gold_start = annotation["qa_start_char"]
    gold_end = annotation.get("qa_end_char") or len(text)
    ambiguous = annotation.get("review_status") == "ambiguous"
    pred = find_qa_start(text)

    if gold_start is None and pred is None:
        return TranscriptResult(id, split, bucket, "correct_abstain", ambiguous=ambiguous)
    if gold_start is None:
        return TranscriptResult(
            id, split, bucket, "false_find", predicted=pred, ambiguous=ambiguous
        )
    if pred is None:
        return TranscriptResult(
            id, split, bucket, "false_abstain", gold_start=gold_start, ambiguous=ambiguous
        )

    sd = sentence_delta(text, pred, gold_start)
    extracted_tokens = _tokens_in(text, pred, len(text))
    gold_tokens = _tokens_in(text, gold_start, gold_end)
    # overlap of extracted [pred, end) with gold [gold_start, gold_end)
    overlap_tokens = _tokens_in(text, max(pred, gold_start), gold_end) if gold_end > pred else 0
    contamination_tokens = _tokens_in(text, pred, gold_start) if pred < gold_start else 0

    return TranscriptResult(
        id=id,
        split=split,
        bucket=bucket,
        outcome="found_vs_gold",
        gold_start=gold_start,
        predicted=pred,
        char_delta=pred - gold_start,
        sentence_delta=sd,
        exact=sd == 0,
        within_tolerance=abs(sd) <= TOLERANCE_SENTENCES,
        precision=overlap_tokens / extracted_tokens if extracted_tokens else None,
        recall=overlap_tokens / gold_tokens if gold_tokens else None,
        contamination=contamination_tokens / extracted_tokens if extracted_tokens else None,
        ambiguous=ambiguous,
    )


@dataclass
class Aggregate:
    n: int = 0
    outcomes: dict = field(default_factory=dict)
    exact: int = 0
    within_tolerance: int = 0
    n_boundary: int = 0  # denominator for exact/tolerance (found_vs_gold)
    sentence_deltas: list = field(default_factory=list)
    precisions: list = field(default_factory=list)
    recalls: list = field(default_factory=list)
    contaminations: list = field(default_factory=list)

    def add(self, r: TranscriptResult) -> None:
        self.n += 1
        self.outcomes[r.outcome] = self.outcomes.get(r.outcome, 0) + 1
        if r.outcome == "found_vs_gold":
            self.n_boundary += 1
            self.exact += bool(r.exact)
            self.within_tolerance += bool(r.within_tolerance)
            self.sentence_deltas.append(r.sentence_delta)
            self.precisions.append(r.precision)
            self.recalls.append(r.recall)
            self.contaminations.append(r.contamination)

    def summary(self) -> dict:
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None

        nb = self.n_boundary
        return {
            "n_transcripts": self.n,
            "outcomes": dict(sorted(self.outcomes.items())),
            "boundary": {
                "denominator": nb,
                "exact": self.exact,
                "exact_rate": self.exact / nb if nb else None,
                f"within_{TOLERANCE_SENTENCES}_sentences": self.within_tolerance,
                "tolerant_rate": self.within_tolerance / nb if nb else None,
                "sentence_deltas": self.sentence_deltas,
            },
            "content": {
                "mean_precision": mean(self.precisions),
                "mean_recall": mean(self.recalls),
                "mean_contamination": mean(self.contaminations),
                "max_contamination": max(
                    (c for c in self.contaminations if c is not None), default=None
                ),
            },
        }
