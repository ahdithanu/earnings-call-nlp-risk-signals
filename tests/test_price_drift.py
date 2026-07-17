import pytest

from src.price_drift import drift_outcomes, forward_return


# 8 consecutive trading days, prices doubling-ish for easy arithmetic.
DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
         "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
CLOSES = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]


def test_forward_return_basic():
    # from index 0 (100) to +3 (130) -> +30%
    assert forward_return(DATES, CLOSES, "2024-01-02", 3) == pytest.approx(30.0)


def test_anchor_rolls_to_next_session():
    # a weekend/holiday anchor lands on the next trading day (2024-01-08 = idx 4)
    r = forward_return(DATES, CLOSES, "2024-01-06", 1)  # 140 -> 150
    assert round(r, 4) == round((150 / 140 - 1) * 100, 4)


def test_insufficient_history_returns_none():
    # not enough sessions ahead
    assert forward_return(DATES, CLOSES, "2024-01-10", 5) is None


def test_zero_baseline_guarded():
    assert forward_return(["2024-01-02", "2024-01-03"], [0.0, 10.0], "2024-01-02", 1) is None


def test_drift_outcomes_shape():
    out = drift_outcomes(DATES, CLOSES, "2024-01-02")
    assert set(out) == {"ret_5d", "ret_fwd_q"}
    assert out["ret_5d"] == pytest.approx(50.0)   # 100 -> 150 (+5 sessions)
    assert out["ret_fwd_q"] is None     # 63 sessions ahead not available
