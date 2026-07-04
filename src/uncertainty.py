"""Negation-aware counting of LM uncertainty terms in earnings-call text."""

import re
from dataclasses import dataclass

# Word tokenizer: keeps internal apostrophes so contractions like "don't"
# survive as single tokens (they matter for negation detection below).
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Negators that can flip an uncertainty term's meaning ("no material risk").
NEGATORS = frozenset(
    {
        "no", "not", "never", "without",
        "don't", "doesn't", "didn't", "wasn't", "isn't", "aren't",
    }
)

# How far back (in tokens) a negator can reach. "no material risk" has the
# negator 2 tokens before the uncertainty term; 3 covers the common cases
# without letting a distant negator suppress an unrelated term.
NEGATION_WINDOW = 3


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with punctuation stripped (apostrophes kept)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class UncertaintyResult:
    total_tokens: int
    uncertainty_count: int
    negation_excluded: int  # matches suppressed by a preceding negator

    @property
    def density(self) -> float | None:
        """Uncertainty terms per 100 tokens; None for empty text."""
        if self.total_tokens == 0:
            return None
        return self.uncertainty_count / self.total_tokens * 100


def count_uncertainty(text: str, lexicon: set[str]) -> UncertaintyResult:
    """Count LM uncertainty terms in ``text``, skipping negated ones.

    Negation rule: a lexicon match is NOT counted if any of the
    NEGATION_WINDOW tokens immediately before it is a negator, so
    "no material risk" does not inflate the score. Suppressed matches are
    tallied separately (``negation_excluded``) so the pipeline can log the
    size of the effect rather than hiding it.
    """
    tokens = tokenize(text)
    count = 0
    excluded = 0
    for i, tok in enumerate(tokens):
        if tok not in lexicon:
            continue
        window = tokens[max(0, i - NEGATION_WINDOW):i]
        if any(w in NEGATORS for w in window):
            excluded += 1
        else:
            count += 1
    return UncertaintyResult(
        total_tokens=len(tokens),
        uncertainty_count=count,
        negation_excluded=excluded,
    )
