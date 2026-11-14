"""Scoring-correctness gate: hand-calculated fixtures vs the implementation.

The expected values in evals/scoring_fixtures.json were computed by hand
from the documented definition (tokenizer, exact lexicon membership,
3-token negation window, per-100-token density) and verified against the
committed lm_uncertainty_terms.txt — not by running the scorer. This test
is therefore an independent check that the implementation computes what
the specification says, including on the quirk cases the fixtures expose
(numerals absent from the denominator, possessive misses, ASCII-only
tokenization, the missing cannot-family negators).

This is a CORRECTNESS gate (implementation == specification). It makes no
claim that a lexicon match expresses real-world uncertainty — contextual
validity needs independent human labels; see evals/README.md.
"""

import json
from pathlib import Path

import pytest

from earnings_signals.lexicon import load_uncertainty_terms
from earnings_signals.uncertainty import count_uncertainty

FIXTURES = json.loads(
    (Path(__file__).parent.parent / "evals" / "scoring_fixtures.json").read_text(encoding="utf-8")
)["fixtures"]

LEXICON = load_uncertainty_terms()


@pytest.mark.parametrize("fx", FIXTURES, ids=[f["id"] for f in FIXTURES])
def test_scoring_fixture(fx):
    result = count_uncertainty(fx["text"], LEXICON)
    exp = fx["expected"]
    assert result.total_tokens == exp["total_tokens"], fx["calculation"]
    assert result.uncertainty_count == exp["uncertainty_count"], fx["calculation"]
    assert result.negation_excluded == exp["negation_excluded"], fx["calculation"]
    if exp["density"] is None:
        assert result.density is None, fx["calculation"]
    else:
        assert result.density == pytest.approx(exp["density"], abs=1e-9), fx["calculation"]
