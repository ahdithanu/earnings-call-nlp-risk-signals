"""Role view: whose Q&A hedging carries the signal — the CEO's or the CFO's?

Splits executive Q&A speech by roster role (src/exec_roles.py) and reruns
the panel spec on each role's uncertainty density. The tone literature
(e.g. CFO-vs-CEO tone studies) suggests CFO language is the more
informative channel; this tests that on our panel.

Coverage caveat, stated up front: roles come from the ``Executives:``
roster that only ~40% of transcripts carry, and a call only enters the
regression sample when BOTH the CEO and the CFO spoke enough in its Q&A.
This is a subsample view, not a full-panel result; the report includes the
full-Q&A benchmark re-estimated on the SAME subsample so the role effects
are compared against a like-for-like baseline, not the headline numbers.

Reads data/processed/tech_uncertainty_features.parquet and writes
results/exec_roles_analysis.txt.
"""

import io

import pandas as pd
import statsmodels.formula.api as smf

PARQUET = "data/processed/tech_uncertainty_features.parquet"
OUT_TXT = "results/exec_roles_analysis.txt"

# Role texts are shorter than the whole Q&A (a CFO may answer only a few
# questions), so the token floor is lower than the main analysis' 500 —
# but nonzero, because density on a couple of sentences is noise.
MIN_ROLE_TOKENS = 200
WINSOR_PCT = 0.01

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n0 = len(df)

    emit("=== coverage (why this is a subsample view) ===")
    emit(f"rows total: {n0}")
    emit(f"role_attributed (titled Executives: roster): {int(df['role_attributed'].sum())} "
         f"({df['role_attributed'].mean() * 100:.1f}%)")
    for role in ("ceo", "cfo"):
        n_any = int(df[f"total_tokens_{role}"].notna().sum())
        n_enough = int((df[f"total_tokens_{role}"] >= MIN_ROLE_TOKENS).sum())
        emit(f"{role.upper()} spoke in Q&A: {n_any}; >= {MIN_ROLE_TOKENS} tokens: {n_enough}")

    df = df[
        (df["total_tokens_ceo"] >= MIN_ROLE_TOKENS)
        & (df["total_tokens_cfo"] >= MIN_ROLE_TOKENS)
        & df["eps_ttm_growth_next_q"].notna()
    ].copy()
    lo, hi = df["eps_ttm_growth_next_q"].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
    df["growth_w"] = df["eps_ttm_growth_next_q"].clip(lo, hi)
    emit(f"\nregression sample (both roles >= {MIN_ROLE_TOKENS} tokens, outcome "
         f"present): {len(df)} rows, {df['ticker'].nunique()} tickers, "
         f"years {df['year'].min()}-{df['year'].max()}")

    emit("\n=== who talks, who hedges ===")
    share = (df["total_tokens_ceo"]
             / (df["total_tokens_ceo"] + df["total_tokens_cfo"]))
    emit(f"CEO share of CEO+CFO Q&A words: median {share.median():.1%}")
    for role in ("ceo", "cfo"):
        emit(f"{role.upper()} uncertainty density: median "
             f"{df[f'uncertainty_density_{role}'].median():.3f}, "
             f"mean {df[f'uncertainty_density_{role}'].mean():.3f}")
    emit(f"corr(ceo density, cfo density): "
         f"{df['uncertainty_density_ceo'].corr(df['uncertainty_density_cfo']):+.3f}")

    # z-score within ticker, as in the main analysis; tickers with too few
    # covered quarters produce NaN/degenerate z and drop out.
    for col in ("uncertainty_density_ceo", "uncertainty_density_cfo",
                "uncertainty_density_qa"):
        z = df.groupby("ticker")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
        df[f"z_{col.rsplit('_', 1)[1]}"] = z
    df = df.dropna(subset=["z_ceo", "z_cfo", "z_qa"])
    kw = dict(cov_type="cluster", cov_kwds={"groups": df["ticker"]})

    emit("\n=== panel: growth_w ~ role density_z + FE (clustered by ticker) ===")
    emit("(benchmark row re-estimates the full-Q&A density on this same subsample)")
    for fe_label, fe in [("ticker FE", "C(ticker)"),
                         ("ticker + quarter FE", "C(ticker) + C(datacqtr)")]:
        emit(f"[{fe_label}]")
        for var, label in [("z_qa", "full-Q&A benchmark"),
                           ("z_ceo", "CEO answers only"),
                           ("z_cfo", "CFO answers only")]:
            fit = smf.ols(f"growth_w ~ {var} + {fe}", data=df).fit(**kw)
            emit(f"  {label:<20} coef = {fit.params[var]:+.3f} pp per 1 SD "
                 f"(t={fit.tvalues[var]:+.2f}, p={fit.pvalues[var]:.3f}, "
                 f"n={int(fit.nobs)})")
        fit = smf.ols(f"growth_w ~ z_ceo + z_cfo + {fe}", data=df).fit(**kw)
        emit(f"  {'CEO + CFO jointly':<20} CEO {fit.params['z_ceo']:+.3f} "
             f"(p={fit.pvalues['z_ceo']:.3f}) | CFO {fit.params['z_cfo']:+.3f} "
             f"(p={fit.pvalues['z_cfo']:.3f})")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
