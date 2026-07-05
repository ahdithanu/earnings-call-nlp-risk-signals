"""Loading of the Loughran-McDonald uncertainty lexicon.

The repo ships the full 297-term LM uncertainty category in
``lm_uncertainty_terms.txt`` (extracted from the LM Master Dictionary,
Uncertainty > 0), so no network access is needed at runtime.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON_PATH = REPO_ROOT / "lm_uncertainty_terms.txt"

# The LM uncertainty category has 297 entries; guard against a truncated
# or accidentally overwritten lexicon file.
EXPECTED_MIN_TERMS = 290


def load_uncertainty_terms(path: Path | str = DEFAULT_LEXICON_PATH) -> set[str]:
    """Return the LM uncertainty terms as a lowercased, deduped set."""
    path = Path(path)
    terms = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(terms) < EXPECTED_MIN_TERMS:
        raise ValueError(
            f"Lexicon at {path} has only {len(terms)} terms; expected the full "
            f"LM uncertainty category (~297). Refusing to run with a partial list."
        )
    return terms
