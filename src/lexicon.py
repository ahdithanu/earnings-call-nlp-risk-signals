"""Loading of Loughran-McDonald sentiment lexicons.

The repo ships two categories from the LM Master Dictionary, so no network
access is needed at runtime:
- ``lm_uncertainty_terms.txt`` — the full 297-term Uncertainty category.
- ``lm_negative_terms.txt`` — the full 2,355-term Negative category, used
  as a tone control (is the signal uncertainty specifically, or just
  bad-news tone?).

Both were extracted from the same dictionary version; the uncertainty list
matches it exactly on all 297 terms.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON_PATH = REPO_ROOT / "lm_uncertainty_terms.txt"
NEGATIVE_LEXICON_PATH = REPO_ROOT / "lm_negative_terms.txt"

# Expected LM category sizes; guard against a truncated or overwritten file.
EXPECTED_MIN_TERMS = 290  # uncertainty (297)
EXPECTED_MIN_NEGATIVE = 2300  # negative (2,355)


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
