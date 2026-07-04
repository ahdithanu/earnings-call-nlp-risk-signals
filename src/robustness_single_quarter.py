"""Robustness check: is the TTM-growth result a smoothing artifact?

The headline outcome (next-quarter TTM EPS growth) is built on windows that
overlap 75% quarter-to-quarter, so its autocorrelation is partly mechanical.
Single-quarter EPS levels are NOT recoverable from this dataset (TTM_t -
TTM_{t-1} = q_t - q_{t-4}, a YoY change, and the level recursion needs four
seed quarters we don't have), so the clean non-overlapping outcome is that
YoY innovation itself:

    eps_yoy_innovation_next_q
        = (TTM_{t+1} - TTM_t) / (|TTM_t| / 4) * 100
        = (q_{t+1} - q_{t-3}) scaled by typical quarterly EPS, in %

Consecutive observations of this outcome share ZERO underlying quarters
(vs 75% overlap for TTM growth), which is exactly the property the
robustness question needs. The quarter-gap guard is inherited from
eps_ttm_next_q, which is NaN unless quarter t+1 is truly adjacent.

Same two specifications as scripts/analyze_uncertainty_growth.py — ticker
FE and ticker+quarter FE, SEs clustered by ticker, predictor z-scored
within ticker, outcomes winsorized 1/99 — run on the COMMON sample where
both outcomes are defined, so the comparison is like-for-like. No other
controls. Within R-squared is computed via Frisch-Waugh-Lovell
(residualize outcome and predictor on the fixed effects, then R-squared
of the residual regression).

Writes results/ttm_vs_single_quarter.csv.

FINDING: the two outcomes are algebraically identical up to scale.
QoQ growth of TTM EPS is (TTM_{t+1} - TTM_t)/|TTM_t| — its numerator is
ALREADY the non-overlapping innovation q_{t+1} - q_{t-3}, so
yoy_innovation = 4 x ttm_growth exactly (corr = 1.0 in the data), and the
regressions match to the digit (identical t, p, within R^2; coefficients
scale by 4). The differencing in "growth" removes the window overlap that
motivated the concern: the smoothing critique applies to TTM levels or to
YoY TTM growth (TTM_t/TTM_{t-4}), not to QoQ growth of TTM. Empirically
the outcome's within-panel AR(1) is 0.05 — no mechanical autocorrelation
to explain the result away.
"""

import pandas as pd
import statsmodels.formula.api as smf

PARQUET = "data/processed/tech_uncertainty_features.parquet"
OUT_CSV = "results/ttm_vs_single_quarter.csv"

MIN_QA_TOKENS = 500
WINSOR_PCT = 0.01

SPECS = [
    ("ticker FE", "C(ticker)"),
    ("ticker + quarter FE", "C(ticker) + C(datacqtr)"),
]


def winsorize(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile([WINSOR_PCT, 1 - WINSOR_PCT])
    return s.clip(lo, hi)


def within_r2(df: pd.DataFrame, y: str, fe: str) -> float:
    """FWL within R^2: residualize y and density_z on the FEs, regress."""
    ry = smf.ols(f"{y} ~ {fe}", data=df).fit().resid
    rx = smf.ols(f"density_z ~ {fe}", data=df).fit().resid
    aux = pd.DataFrame({"ry": ry, "rx": rx})
    return smf.ols("ry ~ rx", data=aux).fit().rsquared


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["qa_isolated"] & (df["total_tokens_qa"] >= MIN_QA_TOKENS)].copy()

    df["density_z"] = df.groupby("ticker")["uncertainty_density_qa"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )

    # Outcome 1: the headline TTM growth (75% overlapping windows).
    df["ttm_growth"] = df["eps_ttm_growth_next_q"]
    # Outcome 2: non-overlapping YoY innovation, % of typical quarterly EPS.
    valid_base = df["eps_ttm"] != 0
    df["yoy_innovation"] = (
        (df["eps_ttm_next_q"] - df["eps_ttm"]) / (df["eps_ttm"].abs() / 4) * 100
    ).where(valid_base)

    common = df.dropna(subset=["density_z", "ttm_growth", "yoy_innovation"]).copy()
    common["ttm_growth_w"] = winsorize(common["ttm_growth"])
    common["yoy_innovation_w"] = winsorize(common["yoy_innovation"])
    print(f"common sample: {len(common)} company-quarters, "
          f"{common['ticker'].nunique()} tickers")

    rows = []
    for outcome, col in [("ttm_growth", "ttm_growth_w"),
                         ("yoy_innovation", "yoy_innovation_w")]:
        for spec_label, fe in SPECS:
            fit = smf.ols(f"{col} ~ density_z + {fe}", data=common).fit(
                cov_type="cluster", cov_kwds={"groups": common["ticker"]}
            )
            rows.append({
                "outcome": outcome,
                "spec": spec_label,
                "coef_density_z": fit.params["density_z"],
                "se_cluster": fit.bse["density_z"],
                "t": fit.tvalues["density_z"],
                "p": fit.pvalues["density_z"],
                "r2_within": within_r2(common, col, fe),
                "n": int(fit.nobs),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("\n" + out.to_string(
        index=False,
        formatters={
            "coef_density_z": "{:+.3f}".format, "se_cluster": "{:.3f}".format,
            "t": "{:+.2f}".format, "p": "{:.4f}".format,
            "r2_within": "{:.4f}".format,
        },
    ))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
