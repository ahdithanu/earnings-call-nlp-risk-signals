"""Steps 2-6: build data/processed/tech_uncertainty_features.parquet.

Pipeline over glopardo/sp500-earnings-transcripts (schema confirmed by
scripts/inspect_dataset.py — 20,681 rows, one `train` split):

  Step 2  Filter to the universe defined in src/universe.py
          (select_universe): GICS Information Technology plus the mega-caps
          GICS files elsewhere (GOOGL, META, AMZN; GOOG excluded as a
          duplicate share class). That module is the single place to change
          which companies are in.
  Step 3  Isolate the Q&A section of each transcript (src/qa_extract.py).
          ~98% of tech transcripts split cleanly; the rest keep only
          full-transcript metrics and qa_isolated=False. Within the Q&A,
          src/qa_isolation.py additionally extracts executive-only answers
          (roster + prepared-remarks speaker attribution) for the _execqa
          metric scope; where attribution fails, _execqa columns are null
          and qa_exec_isolated=False. Where the roster carries titles
          (~40% of rows), src/exec_roles.py further splits exec speech by
          role for the _ceo and _cfo scopes (role_attributed flag).
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

from src.exec_roles import exec_qa_by_role
from src.features import add_next_quarter_eps
from src.lexicon import CONTROL_LOADERS, load_uncertainty_terms
from src.qa_extract import extract_qa
from src.qa_isolation import isolate_executive_qa
from src.uncertainty import count_uncertainty
from src.universe import select_universe

DATASET = "glopardo/sp500-earnings-transcripts"
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


def score_category(text: str | None, lexicon: set[str], cat: str, suffix: str) -> dict:
    """Tone-control metrics for one LM category (negative/litigious/...).

    Uses the same tokenizer and negation handling as the uncertainty score;
    total_tokens is already emitted by ``score``, so only count and density
    are returned here.
    """
    if text is None:
        return {f"{cat}_count_{suffix}": None, f"{cat}_density_{suffix}": None}
    r = count_uncertainty(text, lexicon)
    return {
        f"{cat}_count_{suffix}": r.uncertainty_count,
        f"{cat}_density_{suffix}": r.density,
    }


def main() -> None:
    lexicon = load_uncertainty_terms()
    controls = {cat: loader() for cat, loader in CONTROL_LOADERS.items()}
    df = load_dataset(DATASET)["train"].to_pandas()

    # Step 2: universe filter (src/universe.py is the single source of truth)
    # + panel-key hygiene.
    df = df[select_universe(df)].copy()
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

    # Steps 3-4: Q&A isolation + uncertainty scoring. Three text scopes per
    # transcript: the full call, the Q&A section (all speakers), and the
    # executive-only answers within the Q&A (src/qa_isolation.py) — the
    # _execqa scope strips analyst questions, whose phrasing is itself
    # uncertainty-heavy ("what risks do you see"), out of the signal.
    rows = []
    for transcript in df["transcript"]:
        qa = extract_qa(transcript)
        iso = isolate_executive_qa(transcript)
        exec_qa = iso.text if iso.mode == "exec_turns" else None
        # Role split (src/exec_roles.py) needs the ~40% of transcripts with
        # a titled roster; elsewhere the _ceo/_cfo columns stay null.
        by_role = exec_qa_by_role(transcript)
        ceo_text = (by_role or {}).get("ceo") or None
        cfo_text = (by_role or {}).get("cfo") or None
        row = (
            {"qa_isolated": qa is not None, "qa_exec_isolated": exec_qa is not None,
             "role_attributed": by_role is not None}
            | score(transcript, lexicon, "full")
            | score(qa, lexicon, "qa")
            | score(exec_qa, lexicon, "execqa")
            | score(ceo_text, lexicon, "ceo")
            | score(cfo_text, lexicon, "cfo")
        )
        for cat, lex in controls.items():
            row |= score_category(transcript, lex, cat, "full")
            row |= score_category(qa, lex, cat, "qa")
            row |= score_category(exec_qa, lex, cat, "execqa")
        rows.append(row)
    metrics = pd.DataFrame(rows, index=df.index)
    # Q&A columns have missing values where isolation failed; keep counts as
    # nullable ints and densities as floats instead of object columns.
    count_cols = ["total_tokens", "uncertainty_count", "negation_excluded"]
    count_cols += [f"{c}_count" for c in controls]
    density_cols = ["uncertainty_density_qa", "uncertainty_density_full",
                    "uncertainty_density_execqa"]
    density_cols += [f"{c}_density_qa" for c in controls]
    density_cols += [f"{c}_density_full" for c in controls]
    density_cols += [f"{c}_density_execqa" for c in controls]
    role_count_cols = ["total_tokens", "uncertainty_count", "negation_excluded"]
    metrics = metrics.astype(
        {f"{c}_qa": "Int64" for c in count_cols}
        | {f"{c}_execqa": "Int64" for c in count_cols}
        | {f"{c}_ceo": "Int64" for c in role_count_cols}
        | {f"{c}_cfo": "Int64" for c in role_count_cols}
        | {c: "float64" for c in density_cols}
        | {"uncertainty_density_ceo": "float64", "uncertainty_density_cfo": "float64"}
    )
    out = pd.concat([df[PASSTHROUGH_COLS], metrics], axis=1)
    print(f"qa isolated: {out['qa_isolated'].sum()}/{len(out)} "
          f"({out['qa_isolated'].mean() * 100:.1f}%)")
    print(f"exec-only qa isolated: {out['qa_exec_isolated'].sum()}/{len(out)} "
          f"({out['qa_exec_isolated'].mean() * 100:.1f}%)")
    print(f"role attributed (titled roster): {out['role_attributed'].sum()}/{len(out)} "
          f"({out['role_attributed'].mean() * 100:.1f}%); "
          f"CEO text present: {out['total_tokens_ceo'].notna().sum()}, "
          f"CFO text present: {out['total_tokens_cfo'].notna().sum()}")
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
