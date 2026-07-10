"""Loading of Loughran-McDonald sentiment lexicons.

The repo ships four categories from the LM Master Dictionary, so no network
access is needed at runtime:
- ``lm_uncertainty_terms.txt`` — the full 297-term Uncertainty category (the
  project's predictor).
- ``lm_negative_terms.txt`` — the full 2,355-term Negative category.
- ``lm_litigious_terms.txt`` — the full 904-term Litigious category.
- ``lm_constraining_terms.txt`` — the full 184-term Constraining category.

The last three are tone controls: is the signal uncertainty specifically, or
just some other LM sentiment dimension? All were extracted from the same
dictionary version; the uncertainty list matches it exactly on all 297 terms.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON_PATH = REPO_ROOT / "lm_uncertainty_terms.txt"
NEGATIVE_LEXICON_PATH = REPO_ROOT / "lm_negative_terms.txt"
LITIGIOUS_LEXICON_PATH = REPO_ROOT / "lm_litigious_terms.txt"
CONSTRAINING_LEXICON_PATH = REPO_ROOT / "lm_constraining_terms.txt"

# Expected LM category sizes; guard against a truncated or overwritten file.
EXPECTED_MIN_TERMS = 290  # uncertainty (297)
EXPECTED_MIN_NEGATIVE = 2300  # negative (2,355)
EXPECTED_MIN_LITIGIOUS = 890  # litigious (904)
EXPECTED_MIN_CONSTRAINING = 180  # constraining (184)


def load_lexicon(path: Path | str, expected_min: int) -> set[str]:
    """Return a lexicon file as a lowercased, deduped set of terms.

    Refuses to return a list shorter than ``expected_min`` so a truncated or
    accidentally overwritten file fails loudly instead of silently weakening
    the signal.
    """
    path = Path(path)
    terms = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(terms) < expected_min:
        raise ValueError(
            f"Lexicon at {path} has only {len(terms)} terms; expected at least "
            f"{expected_min}. Refusing to run with a partial list."
        )
    return terms


def load_uncertainty_terms(path: Path | str = DEFAULT_LEXICON_PATH) -> set[str]:
    """Return the LM uncertainty terms as a lowercased, deduped set."""
    return load_lexicon(path, EXPECTED_MIN_TERMS)


def load_negative_terms(path: Path | str = NEGATIVE_LEXICON_PATH) -> set[str]:
    """Return the LM negative terms as a lowercased, deduped set."""
    return load_lexicon(path, EXPECTED_MIN_NEGATIVE)


def load_litigious_terms(path: Path | str = LITIGIOUS_LEXICON_PATH) -> set[str]:
    """Return the LM litigious terms as a lowercased, deduped set."""
    return load_lexicon(path, EXPECTED_MIN_LITIGIOUS)


def load_constraining_terms(path: Path | str = CONSTRAINING_LEXICON_PATH) -> set[str]:
    """Return the LM constraining terms as a lowercased, deduped set."""
    return load_lexicon(path, EXPECTED_MIN_CONSTRAINING)


# Non-predictor tone categories, keyed by the column-name stem the feature
# pipeline uses. Lets build_features score every control category in one loop.
CONTROL_LOADERS = {
    "negative": load_negative_terms,
    "litigious": load_litigious_terms,
    "constraining": load_constraining_terms,
}
