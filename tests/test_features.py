import numpy as np
import pandas as pd

from src.features import add_next_quarter_eps


def make(rows):
    return pd.DataFrame(rows, columns=["ticker", "year", "quarter", "eps"])


def test_forward_shift_within_ticker():
    df = make([("AAPL", 2023, 1, 1.0), ("AAPL", 2023, 2, 1.5), ("AAPL", 2023, 3, 1.2)])
    out = add_next_quarter_eps(df)
    assert out["eps_next_q"].tolist()[:2] == [1.5, 1.2]
    assert np.isnan(out["eps_next_q"].iloc[2])  # last quarter has no successor
    assert out["eps_growth_next_q"].iloc[0] == 50.0


def test_year_boundary_q4_to_q1():
    df = make([("MSFT", 2022, 4, 2.0), ("MSFT", 2023, 1, 2.2)])
    out = add_next_quarter_eps(df)
    assert out["eps_next_q"].iloc[0] == 2.2


def test_gap_in_quarters_leaves_nan():
    # Q2 missing: Q3 must NOT be treated as Q1's next quarter
    df = make([("NVDA", 2023, 1, 1.0), ("NVDA", 2023, 3, 4.0)])
    out = add_next_quarter_eps(df)
    assert out["eps_next_q"].isna().all()


def test_no_leakage_across_tickers():
    df = make([("AAPL", 2023, 1, 1.0), ("MSFT", 2023, 2, 9.0)])
    out = add_next_quarter_eps(df)
    assert out["eps_next_q"].isna().all()


def test_negative_eps_growth_uses_abs_base():
    df = make([("INTC", 2023, 1, -1.0), ("INTC", 2023, 2, -0.5)])
    out = add_next_quarter_eps(df)
    # improvement from -1.0 to -0.5 is +50% on |base|
    assert out["eps_growth_next_q"].iloc[0] == 50.0


def test_zero_eps_base_gives_nan_growth():
    df = make([("AMD", 2023, 1, 0.0), ("AMD", 2023, 2, 0.3)])
    out = add_next_quarter_eps(df)
    assert out["eps_next_q"].iloc[0] == 0.3
    assert np.isnan(out["eps_growth_next_q"].iloc[0])
