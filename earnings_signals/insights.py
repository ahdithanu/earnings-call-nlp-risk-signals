"""Deterministic insight readouts: turn a call's metrics into sentences.

Every surface of the project computes numbers (densities, z-scores,
percentiles); this module is the layer that says what they mean, in plain
English, composed entirely from the data — template-based, unit-testable,
nothing generated that isn't anchored to a computed value.

A readout for a company's latest call contains:
  - the claim: level vs the company's OWN history (percentile, rank,
    "highest since ..."), never a bare z-score
  - the driver: CEO vs CFO hedging vs each of their own norms (the panel
    result says the CEO channel carries the signal, so the readout says
    which channel moved)
  - the character: did uncertainty move alone, or with directional tone —
    distinguishing a hedging spike from a plain bad-news call
  - the receipts: top contributing lexicon terms vs the company's usual
    mix, and the highest-density executive sentences (extracted by
    scripts/build_insights.py)
  - the caveat, always attached: panel-average context, not a forecast

Numbers in, sentences out; no network, no models, no state.
"""

from dataclasses import asdict, dataclass, field

# A move must clear this many standard deviations (vs the company's own
# history) before the readout calls a channel or tone category "elevated".
ELEVATED_Z = 1.0
STRONG_Z = 2.0

CAVEAT = (
    "Panel context: across the S&P 500, one standard deviation of extra "
    "hedging associates with about half a percentage point lower "
    "next-quarter EPS growth on average — a modest, uneven effect. "
    "A high reading is a prompt to read the call, not a forecast."
)


@dataclass
class SeriesContext:
    """A latest value in the context of the company's own prior values."""

    value: float
    z: float | None  # None when history is too short/degenerate
    percentile: float | None  # share of prior calls at or below this value
    n_history: int
    highest_since: str | None  # quarter label of the last prior call >= value


def series_context(history: list[tuple[str, float]], value: float) -> SeriesContext:
    """Context for ``value`` against prior (quarter_label, value) history."""
    values = [v for _, v in history]
    n = len(values)
    if n < 2:
        return SeriesContext(value=value, z=None, percentile=None, n_history=n, highest_since=None)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    z = (value - mean) / var**0.5 if var > 0 else None
    percentile = 100.0 * sum(v <= value for v in values) / n
    highest_since = None
    for label, v in reversed(history):  # most recent prior call first
        if v >= value:
            highest_since = label
            break
    return SeriesContext(
        value=value, z=z, percentile=percentile, n_history=n, highest_since=highest_since
    )


@dataclass
class Excerpt:
    speaker_role: str  # "ceo" | "cfo" | "exec"
    text: str
    matched_terms: list[str]


@dataclass
class Readout:
    ticker: str
    company: str
    quarter: str
    headline: str
    claim: str
    driver: str
    character: str
    terms: str
    excerpts: list[Excerpt] = field(default_factory=list)
    caveat: str = CAVEAT
    # structured values the site/API can render without parsing prose
    density_qa: float | None = None
    z_qa: float | None = None
    percentile_qa: float | None = None
    n_history: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _level_words(ctx: SeriesContext) -> str:
    if ctx.z is None or ctx.percentile is None:
        return "too little history to benchmark"
    if ctx.percentile >= 100.0:
        return f"their highest in {ctx.n_history + 1} calls"
    if ctx.z >= STRONG_Z:
        return f"higher than {ctx.percentile:.0f}% of their prior calls — an unusual spike"
    if ctx.z >= ELEVATED_Z:
        return f"elevated — above {ctx.percentile:.0f}% of their prior calls"
    if ctx.z <= -ELEVATED_Z:
        return f"unusually low — beneath {100 - ctx.percentile:.0f}% of their prior calls"
    return "in their normal range"


def compose_claim(ticker: str, quarter: str, ctx: SeriesContext) -> str:
    base = f"{ticker}'s executives used hedging language at {ctx.value:.2f} per 100 words in the {quarter} Q&A, {_level_words(ctx)}"
    if ctx.highest_since is None and ctx.n_history >= 4:
        return f"{base} ({ctx.n_history}-call history)."
    if ctx.highest_since and ctx.z is not None and ctx.z >= ELEVATED_Z:
        return f"{base}; the last call this hedged was {ctx.highest_since}."
    return f"{base}."


def compose_driver(ceo: SeriesContext | None, cfo: SeriesContext | None) -> str:
    """Which executive channel moved. The CEO channel is the one the panel
    links to outcomes, so a CEO-driven spike is called out as such."""

    def hot(c: SeriesContext | None) -> bool:
        return c is not None and c.z is not None and c.z >= ELEVATED_Z

    if ceo is None and cfo is None:
        return "Speaker roles could not be attributed for this call."
    if hot(ceo) and hot(cfo):
        return (
            "Both the CEO and CFO hedged above their own norms — a broad shift "
            "in the room, and the CEO channel is the one the panel links to "
            "next-quarter outcomes."
        )
    if hot(ceo):
        return (
            "The shift comes from the CEO, not the CFO — the channel the panel "
            "links to next-quarter outcomes."
        )
    if hot(cfo):
        return (
            "The shift comes from the CFO; CEO language stayed in its normal "
            "range. Panel evidence ties outcomes to CEO hedging, so this reads "
            "as a weaker signal."
        )
    return "Neither the CEO nor the CFO moved meaningfully off their own norms."


def compose_character(
    unc: SeriesContext, neg: SeriesContext | None, pos: SeriesContext | None
) -> str:
    """Hedging spike vs bad-news call vs mixed."""
    neg_hot = neg is not None and neg.z is not None and neg.z >= ELEVATED_Z
    pos_cold = pos is not None and pos.z is not None and pos.z <= -ELEVATED_Z
    unc_hot = unc.z is not None and unc.z >= ELEVATED_Z
    if unc_hot and neg_hot:
        return (
            "Uncertainty rose alongside negative tone — this reads as a "
            "bad-news call, not just careful language."
        )
    if unc_hot and pos_cold:
        return (
            "Uncertainty rose while positive tone fell below its norm — "
            "hedging with the optimism drained out."
        )
    if unc_hot:
        return (
            "Uncertainty moved without a matching rise in negative tone — "
            "hedging language specifically, not general negativity."
        )
    return "Tone categories stayed near their usual mix."


def compose_terms(top_terms: list[tuple[str, int]], baseline_terms: list[str]) -> str:
    if not top_terms:
        return "No dominant terms this call."
    new = [t for t, _ in top_terms if t not in baseline_terms]
    listed = ", ".join(f"“{t}”" for t, _ in top_terms[:5])
    if new:
        newly = ", ".join(f"“{t}”" for t in new[:3])
        return f"Driven by {listed}; {newly} {'is' if len(new[:3]) == 1 else 'are'} not part of this company's usual hedging vocabulary."
    return f"Driven by {listed} — consistent with this company's usual hedging vocabulary."


def compose_readout(
    *,
    ticker: str,
    company: str,
    quarter: str,
    qa_history: list[tuple[str, float]],
    qa_value: float,
    ceo: SeriesContext | None,
    cfo: SeriesContext | None,
    negative: SeriesContext | None,
    positive: SeriesContext | None,
    top_terms: list[tuple[str, int]],
    baseline_terms: list[str],
    excerpts: list[Excerpt],
) -> Readout:
    ctx = series_context(qa_history, qa_value)
    if ctx.z is not None and ctx.z >= STRONG_Z:
        headline = f"{company}: executives hedging unusually hard"
    elif ctx.z is not None and ctx.z >= ELEVATED_Z:
        headline = f"{company}: hedging elevated vs own history"
    elif ctx.z is not None and ctx.z <= -ELEVATED_Z:
        headline = f"{company}: unusually direct this quarter"
    else:
        headline = f"{company}: hedging in normal range"
    return Readout(
        ticker=ticker,
        company=company,
        quarter=quarter,
        headline=headline,
        claim=compose_claim(ticker, quarter, ctx),
        driver=compose_driver(ceo, cfo),
        character=compose_character(ctx, negative, positive),
        terms=compose_terms(top_terms, baseline_terms),
        excerpts=excerpts,
        density_qa=round(qa_value, 4),
        z_qa=round(ctx.z, 3) if ctx.z is not None else None,
        percentile_qa=round(ctx.percentile, 1) if ctx.percentile is not None else None,
        n_history=ctx.n_history,
    )
