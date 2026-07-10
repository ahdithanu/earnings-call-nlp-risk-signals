import pytest

from src.lexicon import (
    CONTROL_LOADERS,
    load_constraining_terms,
    load_lexicon,
    load_litigious_terms,
    load_negative_terms,
    load_uncertainty_terms,
)


def test_uncertainty_lexicon_is_full():
    terms = load_uncertainty_terms()
    assert len(terms) == 297  # full LM uncertainty category
    assert all(t == t.lower() for t in terms)
    assert "uncertain" in terms


def test_negative_lexicon_is_full():
    terms = load_negative_terms()
    assert len(terms) == 2355  # full LM negative category
    assert "loss" in terms and "decline" in terms
    # the two categories are distinct lists, not the same file
    assert terms != load_uncertainty_terms()


def test_litigious_and_constraining_are_full():
    assert len(load_litigious_terms()) == 904
    assert len(load_constraining_terms()) == 184


def test_control_loaders_registry():
    # every registered control loads and none is the uncertainty list
    unc = load_uncertainty_terms()
    assert set(CONTROL_LOADERS) == {"negative", "litigious", "constraining"}
    for loader in CONTROL_LOADERS.values():
        terms = loader()
        assert terms and terms != unc


def test_truncated_lexicon_is_rejected(tmp_path):
    partial = tmp_path / "partial.txt"
    partial.write_text("risk\nuncertain\nmaybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least"):
        load_lexicon(partial, expected_min=100)
