import pytest

from earnings_signals.price_providers import (
    PremiumRequired,
    RateLimited,
    parse_alphavantage,
    parse_fmp,
    resolve_provider,
)


def test_parse_fmp_sorts_ascending():
    payload = [  # FMP returns newest-first
        {"symbol": "X", "date": "2024-01-04", "price": 120.0},
        {"symbol": "X", "date": "2024-01-02", "price": 100.0},
        {"symbol": "X", "date": "2024-01-03", "price": 110.0},
    ]
    dates, closes = parse_fmp(payload)
    assert dates == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert closes == [100.0, 110.0, 120.0]


def test_parse_fmp_empty():
    assert parse_fmp([]) == ([], [])
    assert parse_fmp({"Error": "nope"}) == ([], [])


def test_parse_alphavantage_prefers_adjusted_close():
    payload = {
        "Time Series (Daily)": {
            "2024-01-03": {"4. close": "999.0", "5. adjusted close": "110.0"},
            "2024-01-02": {"4. close": "999.0", "5. adjusted close": "100.0"},
        }
    }
    dates, closes = parse_alphavantage(payload)
    assert dates == ["2024-01-02", "2024-01-03"]
    assert closes == [100.0, 110.0]  # adjusted, not the "4. close"


def test_parse_alphavantage_falls_back_to_plain_close():
    payload = {"Time Series (Daily)": {"2024-01-02": {"4. close": "50.0"}}}
    assert parse_alphavantage(payload) == (["2024-01-02"], [50.0])


def test_parse_alphavantage_daily_cap_is_rate_limit_not_premium():
    # the free daily-cap message mentions "premium" too; it must read as a
    # rate limit so the caller stops rather than re-requesting a fallback.
    msg = (
        "We have detected your API key and our standard API rate limit is "
        "25 requests per day. Please subscribe to a premium plan."
    )
    with pytest.raises(RateLimited):
        parse_alphavantage({"Information": msg})


def test_parse_alphavantage_premium_endpoint_gate():
    with pytest.raises(PremiumRequired):
        parse_alphavantage({"Information": "This is a premium endpoint."})


def test_parse_alphavantage_error_and_junk():
    assert parse_alphavantage({"Error Message": "invalid symbol"}) == ([], [])
    assert parse_alphavantage("nonsense") == ([], [])


def test_resolve_provider_auto_prefers_fmp():
    assert resolve_provider("auto", has_fmp=True, has_alphavantage=True) == "fmp"
    assert resolve_provider("auto", has_fmp=False, has_alphavantage=True) == "alphavantage"


def test_resolve_provider_forced():
    assert resolve_provider("alphavantage", has_fmp=True, has_alphavantage=True) == "alphavantage"
    with pytest.raises(ValueError):
        resolve_provider("fmp", has_fmp=False, has_alphavantage=True)
    with pytest.raises(ValueError):
        resolve_provider("alphavantage", has_fmp=True, has_alphavantage=False)


def test_resolve_provider_none_and_unknown():
    with pytest.raises(ValueError):
        resolve_provider("auto", has_fmp=False, has_alphavantage=False)
    with pytest.raises(ValueError):
        resolve_provider("bogus", has_fmp=True, has_alphavantage=True)
