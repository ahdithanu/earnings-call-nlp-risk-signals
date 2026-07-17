"""Post-earnings-call price drift — the return analog of the EPS outcome.

Given a company's daily closing prices and the date of an earnings call, we
measure how the stock moves AFTER the call:

- ``ret_5d``  — close ~5 trading days after the call vs. the first close on or
  after the call (the immediate reaction).
- ``ret_fwd_q`` — close ~63 trading days (≈ one quarter) later vs. that same
  post-call baseline (the drift over the next quarter, the price counterpart of
  next-quarter EPS growth).

Both are computed from a TRADING-day index (positions in the sorted price
series), not calendar days, so weekends/holidays don't distort the horizon.
Pure and unit-tested; the network fetch lives in scripts/fetch_prices.py.
"""

import bisect

REACT_HORIZON = 5    # trading days: immediate post-call reaction
DRIFT_HORIZON = 63   # trading days: ≈ one quarter of drift


def forward_return(dates: list[str], closes: list[float], anchor: str,
                   horizon: int) -> float | None:
    """Percent return from the first close on/after ``anchor`` to ``horizon``
    trading days later. None if the series doesn't reach that far.

    ``dates`` must be sorted ascending and aligned to ``closes`` (ISO date
    strings, lexically sortable). The baseline is the first trading day at or
    after the call date, so a call on a non-trading day anchors to the next
    session.
    """
    i = bisect.bisect_left(dates, anchor)
    j = i + horizon
    if i >= len(dates) or j >= len(dates):
        return None
    p0, p1 = closes[i], closes[j]
    if p0 <= 0:
        return None
    return (p1 / p0 - 1.0) * 100.0


def drift_outcomes(dates: list[str], closes: list[float], anchor: str) -> dict:
    """Both horizons for one call, as a dict (values may be None)."""
    return {
        "ret_5d": forward_return(dates, closes, anchor, REACT_HORIZON),
        "ret_fwd_q": forward_return(dates, closes, anchor, DRIFT_HORIZON),
    }
