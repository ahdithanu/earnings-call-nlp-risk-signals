from src.lexicon import load_uncertainty_terms
from src.uncertainty import count_uncertainty, tokenize

LEX = load_uncertainty_terms()


def test_lexicon_is_full_lm_uncertainty_category():
    assert len(LEX) == 297
    assert {"uncertainty", "risk", "may", "approximately", "volatility"} <= LEX
    # words the old partial list wrongly included are NOT LM uncertainty terms
    assert "if" not in LEX and "hope" not in LEX


def test_tokenize_strips_punctuation_and_keeps_contractions():
    assert tokenize("We don't see risk, per se.") == ["we", "don't", "see", "risk", "per", "se"]


def test_plain_uncertainty_counted():
    r = count_uncertainty("There is significant risk and uncertainty ahead.", LEX)
    assert r.uncertainty_count == 2
    assert r.negation_excluded == 0
    assert r.total_tokens == 7


def test_negated_term_within_window_excluded():
    # "no material risk": negator 2 tokens before the term -> suppressed
    r = count_uncertainty("We see no material risk this quarter.", LEX)
    assert r.uncertainty_count == 0
    assert r.negation_excluded == 1


def test_contraction_negator():
    # both "anticipate" and "volatility" are LM uncertainty terms; the
    # contraction negator suppresses both (each within 3 tokens of "don't")
    r = count_uncertainty("We don't anticipate volatility.", LEX)
    assert r.uncertainty_count == 0
    assert r.negation_excluded == 2


def test_negator_outside_window_does_not_suppress():
    # negator is 4 tokens before "risk" -> outside the 3-token window
    r = count_uncertainty("not that we would call risk", LEX)
    assert r.uncertainty_count == 1
    assert r.negation_excluded == 0


def test_negation_only_reaches_backward():
    # negator AFTER the term must not suppress it
    r = count_uncertainty("risk is not our concern", LEX)
    assert r.uncertainty_count == 1


def test_density_and_empty_text():
    r = count_uncertainty("maybe maybe maybe sure", LEX)
    assert r.density == 75.0
    assert count_uncertainty("", LEX).density is None
