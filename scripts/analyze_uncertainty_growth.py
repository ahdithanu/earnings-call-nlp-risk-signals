"""Weekend 2: does Q&A uncertainty density predict next-quarter EPS growth?

Tests the project's central claim (README "Key Finding") on the full tech
panel instead of a handful of quarters: per-ticker correlations between
Q&A uncertainty density and forward trailing-12M EPS growth, a growth-
context test (do high-growth companies show more-positive correlations
than mature ones?), a ticker fixed-effects panel regression with
ticker-clustered standard errors, and a lead-lag comparison.

Reads data/processed/tech_uncertainty_features.parquet (built by
scripts/build_features.py) and writes:
  results/per_ticker_correlations.csv
  results/uncertainty_growth_analysis.txt

Sample filters: Q&A must be isolated and at least MIN_QA_TOKENS long
(density on a tiny Q&A is noise), and the forward-growth outcome must
exist. Growth is winsorized at 1%/99% for the Pearson/regression results
because TTM EPS near zero produces extreme percentage growth; Spearman
results are reported alongside as the rank-based robustness check.
"""

import io
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PARQUET = "data/processed/tech_uncertainty_features.parquet"
OUT_CSV = "results/per_ticker_correlations.csv"
OUT_TXT = "results/uncertainty_growth_analysis.txt"

MIN_QA_TOKENS = 500
MIN_QUARTERS = 12  # per-ticker correlations need a usable time series
WINSOR_PCT = 0.01

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n0 = len(df)
    df = df[
        df["qa_isolated"]
        & (df["total_tokens_qa"] >= MIN_QA_TOKENS)
        & df["eps_ttm_growth_next_q"].notna()
    ].copy()
    df["density"] = df["uncertainty_density_qa"].astype(float)

    lo, hi = df["eps_ttm_growth_next_q"].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
    df["growth_w"] = df["eps_ttm_growth_next_q"].clip(lo, hi)

    emit("=== sample ===")
    emit(f"rows: {len(df)} of {n0} "
         f"(qa_isolated, total_tokens_qa>={MIN_QA_TOKENS}, growth outcome present)")
    emit(f"tickers: {df['ticker'].nunique()}, years {df['year'].min()}-{df['year'].max()}")
    emit(f"growth winsorized at [{lo:.1f}%, {hi:.1f}%]")

    # ---- per-ticker correlations -------------------------------------------
    rows = []
    for ticker, g in df.groupby("ticker"):
        if len(g) < MIN_QUARTERS:
            continue
        pear_r, pear_p = stats.pearsonr(g["density"], g["growth_w"])
        spear_r, spear_p = stats.spearmanr(g["density"], g["eps_ttm_growth_next_q"])
        rows.append({
            "ticker": ticker,
            "n_quarters": len(g),
            "pearson_r": pear_r,
            "pearson_p": pear_p,
            "spearman_rho": spear_r,
            "spearman_p": spear_p,
            "mean_growth": g["eps_ttm_growth_next_q"].mean(),
            "median_growth": g["eps_ttm_growth_next_q"].median(),
        })
    per = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    os.makedirs("results", exist_ok=True)
    per.to_csv(OUT_CSV, index=False)

    emit("\n=== per-ticker correlations (density_qa vs next-quarter TTM EPS growth) ===")
    emit(f"tickers with >= {MIN_QUARTERS} usable quarters: {len(per)}")
    emit(f"pearson r:  mean {per.pearson_r.mean():+.3f}, median {per.pearson_r.median():+.3f}, "
         f"positive {int((per.pearson_r > 0).sum())}/{len(per)}")
    emit(f"spearman:   mean {per.spearman_rho.mean():+.3f}, median {per.spearman_rho.median():+.3f}, "
         f"positive {int((per.spearman_rho > 0).sum())}/{len(per)}")
    emit(f"significant at p<0.05 (pearson): {int((per.pearson_p < 0.05).sum())}/{len(per)} "
         f"(~{0.05 * len(per):.1f} expected by chance)")
    focus = per[per.ticker.isin(["NVDA", "MSFT"])]
    emit("\nREADME focus tickers (README claimed NVDA r=+0.959, MSFT r=-0.584 on ~3 quarters):")
    emit(focus.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # ---- growth-context test ----------------------------------------------
    # README claim implies a company's growth profile sets the SIGN of its
    # density-growth correlation (high-growth positive, mature negative).
    # If so, per-ticker r should increase with the company's typical growth.
    ctx_r, ctx_p = stats.spearmanr(per["median_growth"], per["pearson_r"])
    emit("\n=== growth-context test ===")
    emit(f"spearman(per-ticker median growth, per-ticker pearson r) = {ctx_r:+.3f} (p={ctx_p:.3f})")
    hi_g = per[per.median_growth >= per.median_growth.median()]
    lo_g = per[per.median_growth < per.median_growth.median()]
    emit(f"high-growth half: mean r {hi_g.pearson_r.mean():+.3f} | "
         f"mature half: mean r {lo_g.pearson_r.mean():+.3f}")

    # ---- fixed-effects panel regression ------------------------------------
    # Density levels differ by company and call style, so the predictor is
    # density z-scored within ticker. Ticker FE absorb level differences in
    # growth; the quarter-FE variant also absorbs market-wide shocks.
    df["density_z"] = df.groupby("ticker")["density"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    df = df.dropna(subset=["density_z"])

    emit("\n=== panel regression: growth_w ~ density_z + FE, SE clustered by ticker ===")
    for label, formula in [
        ("ticker FE", "growth_w ~ density_z + C(ticker)"),
        ("ticker + quarter FE", "growth_w ~ density_z + C(ticker) + C(datacqtr)"),
    ]:
        fit = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["ticker"]}
        )
        emit(f"{label}: coef on density_z = {fit.params['density_z']:+.3f} pp per 1 SD "
             f"(t={fit.tvalues['density_z']:+.2f}, p={fit.pvalues['density_z']:.3f}, "
             f"n={int(fit.nobs)})")

    # ---- lead-lag -----------------------------------------------------------
    # growth_next is the outcome one quarter AHEAD of the call; growth_curr
    # (the same measure shifted within ticker) is growth INTO the quarter
    # being discussed. Comparing the two shows whether density anticipates
    # growth or reacts to it. The quarter-gap guard in the shifted series
    # already leaves NaN across coverage gaps, so a plain shift is safe.
    df = df.sort_values(["ticker", "year", "quarter"])
    df["growth_curr_w"] = df.groupby("ticker")["growth_w"].shift(1)
    emit("\n=== lead-lag (within-ticker density_z) ===")
    for label, col in [("growth into current quarter (reactive)", "growth_curr_w"),
                       ("growth into next quarter (predictive)", "growth_w")]:
        sub = df.dropna(subset=[col])
        r, p = stats.pearsonr(sub["density_z"], sub[col])
        emit(f"{label}: r = {r:+.3f} (p={p:.3f}, n={len(sub)})")

    # ---- negative-tone control ---------------------------------------------
    # Is the signal uncertainty specifically, or just general bad-news tone?
    # Add LM negative density (z-scored within ticker, same tokenizer and
    # negation handling) as a control and check whether the uncertainty
    # coefficient survives. Both models are fit on the SAME sample so the
    # with/without comparison is apples-to-apples.
    df["neg_z"] = df.groupby("ticker")["negative_density_qa"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    df = df.dropna(subset=["neg_z"])
    emit("\n=== negative-tone control "
         "(does uncertainty survive controlling for LM negative tone?) ===")
    emit(f"corr(uncertainty_z, negative_z) in sample: "
         f"{df['density_z'].corr(df['neg_z']):+.3f}")
    for label, fe in [("ticker FE", "C(ticker)"),
                      ("ticker + quarter FE", "C(ticker) + C(datacqtr)")]:
        kw = dict(cov_type="cluster", cov_kwds={"groups": df["ticker"]})
        a = smf.ols(f"growth_w ~ density_z + {fe}", data=df).fit(**kw)
        b = smf.ols(f"growth_w ~ density_z + neg_z + {fe}", data=df).fit(**kw)
        emit(f"\n[{label}] n={int(b.nobs)}")
        emit(f"  uncertainty alone:       {a.params['density_z']:+.3f} pp "
             f"(t={a.tvalues['density_z']:+.2f}, p={a.pvalues['density_z']:.3f})")
        emit(f"  uncertainty + neg ctrl:  {b.params['density_z']:+.3f} pp "
             f"(t={b.tvalues['density_z']:+.2f}, p={b.pvalues['density_z']:.3f})")
        emit(f"  negative tone coef:      {b.params['neg_z']:+.3f} pp "
             f"(t={b.tvalues['neg_z']:+.2f}, p={b.pvalues['neg_z']:.3f})")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_CSV} ({len(per)} tickers) and {OUT_TXT}")


if __name__ == "__main__":
    main()
