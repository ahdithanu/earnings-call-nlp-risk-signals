"""Data-quality gate for the feature parquet.

Run after every build (and by the weekly refresh) so a bad upstream change
fails loudly instead of silently poisoning analyses, the watchlist, and the
site. Checks structure (columns, dtypes), integrity (panel-key uniqueness,
null policies), and plausibility (density ranges, isolation rates, universe
size). Exits non-zero with a list of violations.
"""

import sys

import pandas as pd

PARQUET = "data/processed/sp500_uncertainty_features.parquet"

REQUIRED_COLUMNS = {
    "ticker",
    "company",
    "sector",
    "year",
    "quarter",
    "datacqtr",
    "qa_isolated",
    "qa_exec_isolated",
    "role_attributed",
    "role_source",
    "total_tokens_full",
    "uncertainty_count_full",
    "uncertainty_density_full",
    "total_tokens_qa",
    "uncertainty_density_qa",
    "total_tokens_execqa",
    "uncertainty_density_execqa",
    "total_tokens_ceo",
    "uncertainty_density_ceo",
    "total_tokens_cfo",
    "uncertainty_density_cfo",
    "eps_ttm",
    "eps_ttm_next_q",
    "eps_ttm_growth_next_q",
}

# Plausibility floors chosen loose enough to survive dataset growth but
# tight enough to catch a broken parser or truncated download.
MIN_ROWS = 15_000
MIN_TICKERS = 400
MIN_QA_ISOLATION_RATE = 0.90
MIN_EXEC_ISOLATION_RATE = 0.90
MIN_ROLE_ATTRIBUTION_RATE = 0.85
# LM uncertainty density in earnings-call speech runs ~0.3-1.5 per 100
# tokens; a median outside this band means scoring broke, not markets.
DENSITY_MEDIAN_RANGE = (0.2, 2.0)
DENSITY_HARD_MAX = 25.0


def validate(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
        return problems  # further checks would just cascade

    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (expected >= {MIN_ROWS})")
    if df["ticker"].nunique() < MIN_TICKERS:
        problems.append(f"only {df['ticker'].nunique()} tickers (expected >= {MIN_TICKERS})")

    dupes = df.duplicated(["ticker", "year", "quarter"]).sum()
    if dupes:
        problems.append(f"{dupes} duplicated (ticker, year, quarter) panel keys")

    for col in ("ticker", "year", "quarter", "qa_isolated"):
        n = df[col].isna().sum()
        if n:
            problems.append(f"{n} nulls in required column {col}")

    for flag, col, floor in (
        ("qa_isolated", "qa_isolated", MIN_QA_ISOLATION_RATE),
        ("qa_exec_isolated", "qa_exec_isolated", MIN_EXEC_ISOLATION_RATE),
        ("role_attributed", "role_attributed", MIN_ROLE_ATTRIBUTION_RATE),
    ):
        rate = df[col].mean()
        if rate < floor:
            problems.append(f"{flag} rate {rate:.1%} below floor {floor:.0%}")

    med = df["uncertainty_density_qa"].median()
    lo, hi = DENSITY_MEDIAN_RANGE
    if not (lo <= med <= hi):
        problems.append(f"median Q&A uncertainty density {med:.3f} outside [{lo}, {hi}]")
    dmax = df["uncertainty_density_qa"].max()
    if dmax > DENSITY_HARD_MAX:
        problems.append(f"max Q&A uncertainty density {dmax:.1f} exceeds {DENSITY_HARD_MAX}")

    # isolation-scope consistency: exec text is a subset of the Q&A
    both = df.dropna(subset=["total_tokens_execqa", "total_tokens_qa"])
    bad = (both["total_tokens_execqa"] > both["total_tokens_qa"]).sum()
    if bad:
        problems.append(f"{bad} rows where exec-only tokens exceed Q&A tokens")

    # null-policy consistency: scope metrics are null iff the scope is missing
    mismatch = (df["uncertainty_density_execqa"].isna() != ~df["qa_exec_isolated"]).sum()
    if mismatch:
        problems.append(f"{mismatch} rows where execqa nulls disagree with qa_exec_isolated")

    return problems


def main() -> None:
    df = pd.read_parquet(PARQUET)
    problems = validate(df)
    if problems:
        print(f"FAILED — {len(problems)} data-quality violation(s) in {PARQUET}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(
        f"OK — {PARQUET}: {len(df)} rows, {df['ticker'].nunique()} tickers, "
        f"{df['year'].min()}-{df['year'].max()}, all quality checks passed"
    )


if __name__ == "__main__":
    main()
