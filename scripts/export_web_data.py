"""Generate the self-contained web/index.html for the Uncertainty Explorer.

Renders web/index.template.html — a standalone, vanilla-JS static page — by
injecting the real panel data in place of the ``__EMBEDDED_DATA__`` token, so
the deployed page carries all of its data inline (no build step, no fetch).
The embedded object is ``{panel, companies}`` where each company is
``{ticker, name, quarters:[{label, density, eps|null}], r|null, latestLabel,
latestZ|null}``.

Choices, kept consistent with scripts/analyze_uncertainty_growth.py:
- quarters shown are those with an isolated Q&A of >= 500 tokens (density
  on a tiny or unsplit Q&A is noise, so it is not displayed either);
- eps is next-quarter TTM EPS growth, winsorized at the same panel 1%/99%
  bounds as the analysis; the newest quarter has no outcome yet and is
  emitted as null (the chart renders the density bar without an EPS dot);
- per-company r is the Pearson correlation of the plotted pairs (so the
  number always matches what the viewer sees), null under 12 pairs.
"""

import json

import numpy as np
import pandas as pd

PARQUET = "data/processed/tech_uncertainty_features.parquet"
RECENT = "data/processed/recent_uncertainty_signals.parquet"
PANEL_STATS = "results/panel_stats.json"
TEMPLATE = "web/index.template.html"
OUT = "web/index.html"
DATA_TOKEN = "__EMBEDDED_DATA__"

MIN_QA_TOKENS = 500
MIN_PAIRS_FOR_R = 12
WINSOR_PCT = 0.01


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["qa_isolated"] & (df["total_tokens_qa"] >= MIN_QA_TOKENS)].copy()
    df = df.sort_values(["ticker", "year", "quarter"])

    lo, hi = df["eps_ttm_growth_next_q"].quantile([WINSOR_PCT, 1 - WINSOR_PCT])
    df["eps_w"] = df["eps_ttm_growth_next_q"].clip(lo, hi)

    # Recent quarters (2025Q2+) from the calibrated live source: density-only,
    # no realized EPS yet. Keyed by ticker for append below.
    recent = pd.read_parquet(RECENT).sort_values(["ticker", "year", "quarter"])
    recent_by_ticker = dict(tuple(recent.groupby("ticker")))
    n_recent = 0

    companies = []
    for ticker, g in df.groupby("ticker"):
        quarters = [
            {
                "label": f"Q{int(r.quarter)} '{int(r.year) % 100:02d}",
                "density": round(float(r.uncertainty_density_qa), 2),
                "eps": None if pd.isna(r.eps_w) else round(float(r.eps_w), 1),
            }
            for r in g.itertuples()
        ]
        # append calibrated recent quarters (already strictly newer than the
        # panel's last quarter for this ticker; EPS not yet realized)
        for r in recent_by_ticker.get(ticker, pd.DataFrame()).itertuples():
            quarters.append({
                "label": f"Q{int(r.quarter)} '{int(r.year) % 100:02d}",
                "density": round(float(r.uncertainty_density_qa), 2),
                "eps": None,
            })
            n_recent += 1
        # r is computed only on quarters with a realized EPS outcome, so the
        # density-only recent quarters do not affect it
        pairs = g.dropna(subset=["eps_w"])
        r_val = (
            round(float(np.corrcoef(pairs["uncertainty_density_qa"], pairs["eps_w"])[0, 1]), 2)
            if len(pairs) >= MIN_PAIRS_FOR_R
            else None
        )
        # "hedging now" signal for the watchlist: the latest quarter's Q&A
        # density z-scored against this company's PRIOR quarters only (same
        # out-of-sample logic as scripts/latest_signals.py). None until a
        # company has enough history to define a norm.
        dens = [q["density"] for q in quarters]
        latest_z = None
        if len(dens) > MIN_PAIRS_FOR_R:
            prior = np.array(dens[:-1], dtype=float)
            sd = prior.std(ddof=0)
            if sd > 0:
                latest_z = round(float((dens[-1] - prior.mean()) / sd), 2)
        companies.append({
            "ticker": ticker,
            "name": g["company"].iloc[-1],
            "quarters": quarters,
            "r": r_val,
            "latestLabel": quarters[-1]["label"],
            "latestZ": latest_z,
        })

    # PANEL is derived, never hand-typed: the regression stats come from the
    # analysis (results/panel_stats.json), and the ticker count from the data
    # here. The explorer reads these, so a universe change needs no HTML edits.
    with open(PANEL_STATS, encoding="utf-8") as f:
        stats = json.load(f)
    panel = {
        "coefLow": stats["coefLow"],
        "coefHigh": stats["coefHigh"],
        "n": stats["n"],
        "tickers": len(companies),
        "p": stats["p"],
    }

    # Inject the data inline into the standalone template. json.dumps is safe
    # to embed in a <script>, except for "</" which could close the tag early.
    data_json = json.dumps({"panel": panel, "companies": companies},
                           separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")
    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()
    if DATA_TOKEN not in template:
        raise ValueError(f"{TEMPLATE} is missing the {DATA_TOKEN} injection token")
    html = template.replace(DATA_TOKEN, data_json)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    n_q = sum(len(c["quarters"]) for c in companies)
    print(f"wrote {OUT}: {len(companies)} companies, {n_q} quarters "
          f"({n_recent} recent density-only), eps winsorized at [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
