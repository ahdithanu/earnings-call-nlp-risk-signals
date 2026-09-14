from earnings_signals.insights import (
    Excerpt,
    SeriesContext,
    compose_character,
    compose_driver,
    compose_readout,
    compose_terms,
    series_context,
)

HISTORY = [(f"20{13 + i // 4}Q{i % 4 + 1}", 0.5 + 0.02 * (i % 5)) for i in range(20)]


def ctx(z: float | None) -> SeriesContext:
    return SeriesContext(value=1.0, z=z, percentile=90.0, n_history=20, highest_since=None)


def test_series_context_spike():
    c = series_context(HISTORY, 1.2)
    assert c.z is not None and c.z > 2
    assert c.percentile == 100.0
    assert c.highest_since is None  # nothing prior was this high
    assert c.n_history == 20


def test_series_context_normal_and_short_history():
    c = series_context(HISTORY, 0.54)
    assert c.z is not None and abs(c.z) < 1
    assert c.highest_since is not None  # some prior call was >= this
    short = series_context(HISTORY[:1], 0.9)
    assert short.z is None and short.percentile is None


def test_driver_attribution_ceo_vs_cfo():
    assert "comes from the CEO" in compose_driver(ctx(1.8), ctx(0.1))
    weaker = compose_driver(ctx(0.0), ctx(2.2))
    assert "comes from the CFO" in weaker and "weaker signal" in weaker
    assert "Both the CEO and CFO" in compose_driver(ctx(1.5), ctx(1.5))
    assert "could not be attributed" in compose_driver(None, None)
    assert "Neither" in compose_driver(ctx(0.2), ctx(-0.3))


def test_character_distinguishes_hedging_from_bad_news():
    assert "bad-news call" in compose_character(ctx(1.5), ctx(1.5), ctx(0.0))
    assert "not general negativity" in compose_character(ctx(1.5), ctx(0.0), ctx(0.0))
    assert "optimism drained" in compose_character(ctx(1.5), ctx(0.0), ctx(-1.5))
    assert "usual mix" in compose_character(ctx(0.0), ctx(0.0), ctx(0.0))


def test_terms_flags_new_vocabulary():
    s = compose_terms([("tariffs", 9), ("could", 7)], baseline_terms=["could", "risk"])
    assert "tariffs" in s and "not part of this company's usual hedging vocabulary" in s
    usual = compose_terms([("could", 7)], baseline_terms=["could"])
    assert "consistent with" in usual
    assert compose_terms([], []) == "No dominant terms this call."


def test_full_readout_composition():
    r = compose_readout(
        ticker="ACME",
        company="Acme Corp",
        quarter="2025Q1",
        qa_history=HISTORY,
        qa_value=1.2,
        ceo=ctx(2.1),
        cfo=ctx(0.2),
        negative=ctx(0.1),
        positive=ctx(-0.2),
        top_terms=[("uncertain", 8), ("could", 6)],
        baseline_terms=["could"],
        excerpts=[
            Excerpt(
                speaker_role="ceo",
                text="It could be volatile.",
                matched_terms=["could", "volatile"],
            )
        ],
    )
    assert "hedging unusually hard" in r.headline
    assert "highest in 21 calls" in r.claim
    assert "comes from the CEO" in r.driver
    assert "not general negativity" in r.character
    assert r.z_qa is not None and r.z_qa > 2
    assert r.caveat.endswith("not a forecast.")
    d = r.to_dict()
    assert d["excerpts"][0]["matched_terms"] == ["could", "volatile"]
