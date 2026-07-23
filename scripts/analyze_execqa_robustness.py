"""Robustness: does the uncertainty signal strengthen on executive-only speech?

The main analysis (scripts/analyze_uncertainty_growth.py) scores the whole
Q&A section, which mixes executive hedging with analyst phrasing — and
analyst questions are themselves uncertainty-heavy ("what risks do you
see"). The _execqa columns (src/qa_isolation.py) strip analyst speech out.
If the hypothesis is right that the signal lives in *executive* language,
the exec-only measure should carry at least as much signal as the full-Q&A
measure, on identical rows.

Design: every comparison runs on the SAME sample (rows where both scopes
are isolated and long enough), with the same winsorization, z-scoring, FE
specs, and clustered SEs as the main analysis. A final horse-race puts both
z-scored densities in one regression to see which dominates.

Reads data/processed/tech_uncertainty_features.parquet and writes
results/execqa_robustness.txt.
"""

import io

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PARQUET = "data/processed/tech_uncertainty_features.parquet"
OUT_TXT = "results/execqa_robustness.txt"

# Same thresholds as the main analysis; the token floor applies to BOTH
# scopes so the sample is identical for the comparison.
MIN_TOKENS = 500
MIN_QUARTERS = 12
WINSOR_PCT = 0.01

SCOPES = {"qa": "full Q&A (all speakers)", "execqa": "executive answers only"}

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n0 = len(df)
    df = df[
        df["qa_isolated"]
        & df["qa_exec_isolated"]
        & (df["total_tokens_qa"] >= MIN_TOKENS)
        & (df["total_tokens_execqa"] >= MIN_TOKENS)
        & df["eps_ttm_growth_next_q"].notna()
    ].copy()

    lo, hi = df["eps_ttm_growth_next_q"].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
    df["growth_w"] = df["eps_ttm_growth_next_q"].clip(lo, hi)

    for scope in SCOPES:
        df[f"z_{scope}"] = (
            df.groupby("ticker")[f"uncertainty_density_{scope}"]
        ).transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    df = df.dropna(subset=[f"z_{s}" for s in SCOPES])

    emit("=== sample (identical rows for both scopes) ===")
    emit(f"rows: {len(df)} of {n0} (both scopes isolated, both >= {MIN_TOKENS} "
         f"tokens, growth outcome present)")
    emit(f"tickers: {df['ticker'].nunique()}, years {df['year'].min()}-{df['year'].max()}")
    emit(f"corr(density_qa, density_execqa): "
         f"{df['uncertainty_density_qa'].corr(df['uncertainty_density_execqa']):+.3f}")
    emit(f"exec share of Q&A tokens: median "
         f"{(df['total_tokens_execqa'] / df['total_tokens_qa']).median():.1%}")

    kw = dict(cov_type="cluster", cov_kwds={"groups": df["ticker"]})

    # ---- per-scope: panel FE regressions + per-ticker correlation summary --
    for scope, label in SCOPES.items():
        emit(f"\n=== scope: {label} [{scope}] ===")
        for fe_label, fe in [("ticker FE", "C(ticker)"),
                             ("ticker + quarter FE", "C(ticker) + C(datacqtr)")]:
            fit = smf.ols(f"growth_w ~ z_{scope} + {fe}", data=df).fit(**kw)
            emit(f"  [{fe_label}] coef = {fit.params[f'z_{scope}']:+.3f} pp per 1 SD "
                 f"(t={fit.tvalues[f'z_{scope}']:+.2f}, "
                 f"p={fit.pvalues[f'z_{scope}']:.3f}, n={int(fit.nobs)})")

        rs = []
        for _, g in df.groupby("ticker"):
            if len(g) >= MIN_QUARTERS:
                rs.append(stats.pearsonr(
                    g[f"uncertainty_density_{scope}"], g["growth_w"])[0])
        rs = pd.Series(rs)
        emit(f"  per-ticker pearson r (>= {MIN_QUARTERS} quarters, {len(rs)} tickers): "
             f"mean {rs.mean():+.3f}, median {rs.median():+.3f}, "
             f"positive {int((rs > 0).sum())}/{len(rs)}")

        sub = df.sort_values(["ticker", "year", "quarter"])
        curr = sub.groupby("ticker")["growth_w"].shift(1)
        both = pd.DataFrame({"z": sub[f"z_{scope}"], "curr": curr}).dropna()
        r_pred, p_pred = stats.pearsonr(df[f"z_{scope}"], df["growth_w"])
        r_reac, p_reac = stats.pearsonr(both["z"], both["curr"])
        emit(f"  lead-lag: predictive r={r_pred:+.3f} (p={p_pred:.3f}) vs "
             f"reactive r={r_reac:+.3f} (p={p_reac:.3f})")

    # ---- horse race: both scopes in one model ------------------------------
    # With both z-scored densities included, whichever scope carries the
    # signal keeps its coefficient; the contaminated one should shrink.
    emit("\n=== horse race: growth_w ~ z_qa + z_execqa + FE ===")
    for fe_label, fe in [("ticker FE", "C(ticker)"),
                         ("ticker + quarter FE", "C(ticker) + C(datacqtr)")]:
        fit = smf.ols(f"growth_w ~ z_qa + z_execqa + {fe}", data=df).fit(**kw)
        emit(f"[{fe_label}] n={int(fit.nobs)}: "
             f"z_qa {fit.params['z_qa']:+.3f} (p={fit.pvalues['z_qa']:.3f}) | "
             f"z_execqa {fit.params['z_execqa']:+.3f} "
             f"(p={fit.pvalues['z_execqa']:.3f})")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
