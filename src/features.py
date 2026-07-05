"""Panel feature construction: forward-shifted EPS outcome variables."""

import numpy as np
import pandas as pd


def add_next_quarter_eps(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``eps_next_q`` and ``eps_growth_next_q`` to a company-quarter panel.

    Expects columns: ticker, year, quarter, eps. Rows are sorted per ticker by
    (year, quarter) and EPS is shifted back one row, but the shifted value is
    only kept when that next row really is the immediately following calendar
    quarter — a gap in coverage (e.g. Q2 missing) leaves NaN rather than
    silently treating Q3 as "next quarter" of Q1.
    """
    df = df.sort_values(["ticker", "year", "quarter"]).reset_index(drop=True)

    # Quarter index on a single integer axis so "next quarter" is +1 exactly.
    qidx = df["year"] * 4 + (df["quarter"] - 1)
    next_eps = df.groupby("ticker")["eps"].shift(-1)
    next_qidx = qidx.groupby(df["ticker"]).shift(-1)

    gap = next_qidx - qidx
    df["eps_next_q"] = next_eps.where(gap == 1)

    # QoQ % growth into next quarter. Undefined when this quarter's EPS is 0
    # (division by zero) — left as NaN, not fabricated.
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (df["eps_next_q"] - df["eps"]) / df["eps"].abs() * 100
    df["eps_growth_next_q"] = growth.where(df["eps"] != 0)
    return df
