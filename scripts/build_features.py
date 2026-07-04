"""Steps 2-6: build data/processed/tech_uncertainty_features.parquet.

Pipeline over glopardo/sp500-earnings-transcripts (schema confirmed by
scripts/inspect_dataset.py — 20,681 rows, one `train` split):

  Step 2  Filter to the tech universe. The dataset HAS a GICS sector field,
          so the preferred path applies: sector == "Information Technology"
          (2,919 rows, 69 tickers, 2013-2025), unioned with
          EXTRA_TECH_TICKERS (GOOGL, META, AMZN — mega-caps GICS files
          under other sectors; GOOG excluded as a duplicate share class).
          The hardcoded TECH_TICKERS list in src/universe.py stays a
          fallback only.
  Step 3  Isolate the Q&A section of each transcript (src/qa_extract.py).
          ~96% of tech transcripts split cleanly; the rest keep only
          full-transcript metrics and qa_isolated=False.
  Step 4  Negation-aware LM uncertainty counts (src/uncertainty.py) on the
          full transcript and on the Q&A section separately.
  Step 5  Forward EPS outcome (src/features.py). The dataset has no single-
          quarter EPS column; the realized-earnings series is the trailing
          12-month EPS at end of quarter (eps12mtrailing_eoq), so that is
          the panel's `eps`. Output columns are renamed with an eps_ttm_
          prefix so the trailing-12M basis is explicit. Rows with a missing
          year/quarter key (163 in the raw data) cannot be panel-indexed
          and are dropped; duplicate ticker-quarters keep the longest
          transcript.
  Step 6  Write the feature table (ids + metrics, no transcript text) to
          data/processed/tech_uncertainty_features.parquet.

Requires network access to huggingface.co on first run (cached after).
"""

import os

# Restricted-egress environments may allow huggingface.co but not the Xet
# CDN hosts (*.xethub.hf.co); the classic download path works everywhere.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from datasets import load_dataset

from src.features import add_next_quarter_eps
from src.lexicon import load_uncertainty_terms
from src.qa_extract import extract_qa
from src.uncertainty import count_uncertainty
from src.universe import EXTRA_TECH_TICKERS

DATASET = "glopardo/sp500-earnings-transcripts"
TECH_SECTOR = "Information Technology"
OUT_PATH = "data/processed/tech_uncertainty_features.parquet"

PASSTHROUGH_COLS = [
    "ticker", "company", "sector", "industry", "cik",
    "year", "quarter", "datacqtr", "datafqtr", "earnings_date",
    "eps12mtrailing_qavg", "eps12mtrailing_eoq",
    "eps12mfwd_qavg", "eps12mfwd_eoq", "eps_lt",
    "peforw_qavg", "peforw_eoq",
]


def score(text: str | None, lexicon: set[str], suffix: str) -> dict:
    """Uncertainty metrics for one text, with column-name suffix applied."""
    if text is None:
        return {
            f"total_tokens_{suffix}": None,
            f"uncertainty_count_{suffix}": None,
            f"negation_excluded_{suffix}": None,
            f"uncertainty_density_{suffix}": None,
        }
    r = count_uncertainty(text, lexicon)
    return {
        f"total_tokens_{suffix}": r.total_tokens,
        f"uncertainty_count_{suffix}": r.uncertainty_count,
        f"negation_excluded_{suffix}": r.negation_excluded,
        f"uncertainty_density_{suffix}": r.density,
    }


def main() -> None:
    lexicon = load_uncertainty_terms()
    df = load_dataset(DATASET)["train"].to_pandas()

    # Step 2: tech filter + panel-key hygiene. GICS Information Technology
    # plus the mega-caps GICS files elsewhere (GOOGL, META, AMZN).
    df = df[
        (df["sector"] == TECH_SECTOR) | df["ticker"].isin(EXTRA_TECH_TICKERS)
    ].copy()
    n_tech = len(df)
    df = df.dropna(subset=["year", "quarter"])
    print(f"tech rows: {n_tech} ({n_tech - len(df)} dropped for missing year/quarter)")
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)

    df["_tlen"] = df["transcript"].str.len()
    df = (
        df.sort_values("_tlen", ascending=False)
        .drop_duplicates(["ticker", "year", "quarter"])
        .drop(columns="_tlen")
    )
    print(f"after dedup on (ticker, year, quarter): {len(df)}")

    # Steps 3-4: Q&A isolation + uncertainty scoring.
    rows = []
    for transcript in df["transcript"]:
        qa = extract_qa(transcript)
        rows.append(
            {"qa_isolated": qa is not None}
            | score(transcript, lexicon, "full")
            | score(qa, lexicon, "qa")
        )
    metrics = pd.DataFrame(rows, index=df.index)
    # Q&A columns have missing values where isolation failed; keep counts as
    # nullable ints and densities as floats instead of object columns.
    metrics = metrics.astype(
        {f"{c}_qa": "Int64" for c in ("total_tokens", "uncertainty_count", "negation_excluded")}
        | {"uncertainty_density_qa": "float64", "uncertainty_density_full": "float64"}
    )
    out = pd.concat([df[PASSTHROUGH_COLS], metrics], axis=1)
    print(f"qa isolated: {out['qa_isolated'].sum()}/{len(out)} "
          f"({out['qa_isolated'].mean() * 100:.1f}%)")
    print(f"total negation-suppressed matches (full): "
          f"{out['negation_excluded_full'].sum()}")

    # Step 5: forward EPS outcome on the trailing-12M series.
    out["eps"] = out["eps12mtrailing_eoq"]
    out = add_next_quarter_eps(out)
    out = out.rename(columns={
        "eps": "eps_ttm",
        "eps_next_q": "eps_ttm_next_q",
        "eps_growth_next_q": "eps_ttm_growth_next_q",
    })
    print(f"eps_ttm_next_q coverage: {out['eps_ttm_next_q'].notna().sum()}/{len(out)}")

    # Step 6: write parquet.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {len(out)} rows x {len(out.columns)} cols, "
          f"{out['ticker'].nunique()} tickers, "
          f"{out['year'].min()}-{out['year'].max()}")


if __name__ == "__main__":
    main()
