"""Build insight readouts for every company's latest panel call.

One pass over the transcript corpus + the committed feature parquet:

  parquet  -> per-ticker history series (Q&A / CEO / CFO / tone densities)
  corpus   -> evidence for the LATEST call per ticker: which lexicon terms
              drove the score (vs the company's prior-call term mix) and
              the highest-density executive sentences, with matched terms
  compose  -> earnings_signals/insights.py turns both into plain-English
              readouts (claim / driver / character / terms / receipts)

Writes data/processed/latest_insights.json — the artifact behind
GET /insight/{ticker}, the explorer's readout cards, and the weekly brief.
Only brief excerpt sentences are stored, never transcripts.
"""

import json
import os
import re
from collections import Counter
from datetime import UTC, datetime

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from datasets import load_dataset

from earnings_signals.exec_roles import exec_qa_by_role
from earnings_signals.insights import Excerpt, compose_readout, series_context
from earnings_signals.lexicon import load_uncertainty_terms
from earnings_signals.qa_extract import extract_qa
from earnings_signals.qa_isolation import isolate_executive_qa
from earnings_signals.uncertainty import NEGATION_WINDOW, NEGATORS, tokenize
from earnings_signals.universe import select_universe

DATASET = "glopardo/sp500-earnings-transcripts"
PARQUET = "data/processed/sp500_uncertainty_features.parquet"
OUT_PATH = "data/processed/latest_insights.json"

TOP_TERMS = 5
BASELINE_TERMS = 10
MAX_EXCERPTS = 3
MIN_EXCERPT_TOKENS = 8
MAX_EXCERPT_TOKENS = 70
MIN_EXCERPT_MATCHES = 2

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def term_counts(text: str, lexicon: set[str]) -> Counter:
    """Negation-aware term counts (same rule as the density scorer)."""
    tokens = tokenize(text)
    counts: Counter = Counter()
    for i, tok in enumerate(tokens):
        if tok in lexicon and not any(
            w in NEGATORS for w in tokens[max(0, i - NEGATION_WINDOW) : i]
        ):
            counts[tok] += 1
    return counts


def best_sentences(text: str, role: str, lexicon: set[str]) -> list[Excerpt]:
    """Highest-uncertainty sentences of one speaker scope, with matches."""
    out = []
    for m in _SENTENCE_RE.finditer(text):
        sentence = m.group(0).strip()
        tokens = tokenize(sentence)
        if not (MIN_EXCERPT_TOKENS <= len(tokens) <= MAX_EXCERPT_TOKENS):
            continue
        matched = [t for t in dict.fromkeys(tokens) if t in lexicon]
        n_matches = sum(1 for t in tokens if t in lexicon)
        if n_matches >= MIN_EXCERPT_MATCHES:
            out.append((n_matches / len(tokens), Excerpt(role, sentence, matched)))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return [e for _, e in out]


def history_series(g: pd.DataFrame, col: str) -> list[tuple[str, float]]:
    prior = g.iloc[:-1]
    return [
        (str(r.datacqtr), float(getattr(r, col)))
        for r in prior.itertuples()
        if pd.notna(getattr(r, col))
    ]


def latest_context(g: pd.DataFrame, col: str):
    latest_val = g.iloc[-1][col]
    if pd.isna(latest_val):
        return None
    return series_context(history_series(g, col), float(latest_val))


def main() -> None:
    lexicon = load_uncertainty_terms()
    panel = pd.read_parquet(PARQUET).sort_values(["ticker", "year", "quarter"])

    raw = load_dataset(DATASET)["train"].to_pandas()
    raw = raw[select_universe(raw) & raw["year"].notna()].copy()
    raw["year"] = raw["year"].astype(int)
    raw["quarter"] = raw["quarter"].astype(int)
    raw["_tlen"] = raw["transcript"].str.len()
    raw = raw.sort_values("_tlen", ascending=False).drop_duplicates(["ticker", "year", "quarter"])
    raw = raw.sort_values(["ticker", "year", "quarter"])

    insights: dict[str, dict] = {}
    skipped = 0
    for ticker, g in panel.groupby("ticker"):
        latest = g.iloc[-1]
        if pd.isna(latest["uncertainty_density_qa"]) or len(g) < 3:
            skipped += 1
            continue

        calls = raw[raw["ticker"] == ticker]
        latest_call = calls[
            (calls["year"] == latest["year"]) & (calls["quarter"] == latest["quarter"])
        ]
        if latest_call.empty:
            skipped += 1
            continue
        transcript = latest_call.iloc[0]["transcript"]

        # --- evidence: term attribution --------------------------------------
        qa = extract_qa(transcript) or ""
        latest_terms = term_counts(qa, lexicon)
        baseline: Counter = Counter()
        for t in calls[
            ~((calls["year"] == latest["year"]) & (calls["quarter"] == latest["quarter"]))
        ]["transcript"]:
            baseline.update(term_counts(extract_qa(t) or "", lexicon))
        top_terms = latest_terms.most_common(TOP_TERMS)
        baseline_terms = [t for t, _ in baseline.most_common(BASELINE_TERMS)]

        # --- evidence: excerpts (prefer role-attributed speech) ---------------
        by_role, _src = exec_qa_by_role(transcript)
        excerpts: list[Excerpt] = []
        if by_role:
            for role in ("ceo", "cfo"):
                if by_role.get(role):
                    excerpts.extend(best_sentences(by_role[role], role, lexicon)[:2])
        if not excerpts:
            iso = isolate_executive_qa(transcript)
            source = iso.text if iso.mode == "exec_turns" else qa
            excerpts = best_sentences(source, "exec", lexicon)
        excerpts = excerpts[:MAX_EXCERPTS]

        readout = compose_readout(
            ticker=str(ticker),
            company=str(latest["company"]),
            quarter=str(latest["datacqtr"]),
            qa_history=history_series(g, "uncertainty_density_qa"),
            qa_value=float(latest["uncertainty_density_qa"]),
            ceo=latest_context(g, "uncertainty_density_ceo"),
            cfo=latest_context(g, "uncertainty_density_cfo"),
            negative=latest_context(g, "negative_density_qa"),
            positive=latest_context(g, "positive_density_qa"),
            top_terms=[(t, int(c)) for t, c in top_terms],
            baseline_terms=baseline_terms,
            excerpts=excerpts,
        )
        insights[str(ticker)] = readout.to_dict()

    payload = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "source": "panel latest call per ticker",
        "count": len(insights),
        "insights": insights,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    size_kb = os.path.getsize(OUT_PATH) // 1024
    print(f"wrote {OUT_PATH}: {len(insights)} readouts ({skipped} skipped), {size_kb} KB")


if __name__ == "__main__":
    main()
