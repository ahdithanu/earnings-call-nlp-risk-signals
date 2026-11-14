"""Build the extraction-benchmark manifest: select 30 real transcripts.

One-time, seeded, reproducible selection. The committed output
(evals/manifest.json) is the benchmark's source of truth; this script
documents HOW the 30 were chosen and can regenerate the same selection
from the pinned dataset revisions.

Selection targets DIFFICULTY COVERAGE, not a random sample — and the
difficulty signals are computed with the extractor's own regexes (which
tier fires, how many header mentions, abstention), so the benchmark is
deliberately enriched for hard cases. That is selection bias, disclosed:
aggregate numbers from this set do not estimate corpus-wide accuracy.
Gold annotations are NOT taken from the extractor — they are written by
reading each transcript (see evals/ANNOTATION_GUIDE.md).

Buckets (glopardo = Seeking Alpha-style panel source, rogersurf = Motley
Fool-style live source):
  abstain        extractor finds no boundary (genuinely-missing-Q&A and
                 missed-boundary candidates)
  header_multi   explicit Q&A header AND >=2 header-phrase mentions
                 (intro announcement + real header — the repeated-mention trap)
  header_single  explicit header, single mention
  firstq         no in-window header; "first question" cue fires
  handoff        neither header nor first-question cue; operator-handoff
                 fallback fires (analyst handoffs)
  extreme        shortest / longest transcripts (position-window stress)
  rs_mf          rogersurf, Motley Fool name/--/title turn format
  rs_colon       rogersurf, colon "Name:" format
  rs_abstain     rogersurf, extractor abstains

Constraints: unique ticker across the whole benchmark (kills identical /
near-duplicate risk, including the same call appearing in both sources),
>= 7 GICS sectors, years spanning the panel and the live source. The
20/10 dev/holdout split is assigned HERE, before annotation, stratified
by bucket, so every case type appears on both sides and the split can't
react to labels.

Usage: python -m evals.build_manifest   (writes evals/manifest.json)
"""

import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from earnings_signals.qa_extract import _FIRST_QUESTION_RE, _QA_HEADER_RE, find_qa_start
from earnings_signals.universe import select_universe

PANEL_DS = "glopardo/sp500-earnings-transcripts"
RECENT_DS = "Rogersurf/earnings-call-transcripts"
PANEL_PARQUET = "data/processed/sp500_uncertainty_features.parquet"
OUT = Path("evals/manifest.json")
SEED = 20261114
MIN_CHARS = 3000  # below this a "transcript" is usually a stub page

_MF_HEADER_RE = re.compile(r"\n[^\n]{2,60}\n--\n[^\n]{2,90}\n")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def signals(text: str) -> dict:
    n = len(text)
    start = find_qa_start(text)
    header_mentions = len(_QA_HEADER_RE.findall(text))
    in_window_header = any(0.15 * n <= m.start() <= 0.95 * n for m in _QA_HEADER_RE.finditer(text))
    in_window_firstq = any(
        0.15 * n <= m.start() <= 0.95 * n for m in _FIRST_QUESTION_RE.finditer(text)
    )
    if start is None:
        tier = "abstain"
    elif (
        in_window_header
        and text[start:].lower().startswith(("question", "question-"))
        or in_window_header
    ):
        tier = "header"
    elif in_window_firstq:
        tier = "firstq"
    else:
        tier = "handoff"
    return {
        "tier": tier,
        "header_mentions": header_mentions,
        "n_chars": n,
        "mf_format": len(_MF_HEADER_RE.findall(text)) >= 5,
    }


def pick(rows: list[dict], k: int, rng: random.Random, used_tickers: set[str]) -> list[dict]:
    rng.shuffle(rows)
    out = []
    for r in rows:
        if r["ticker"] in used_tickers:
            continue
        out.append(r)
        used_tickers.add(r["ticker"])
        if len(out) == k:
            break
    return out


def main() -> None:
    api = HfApi()
    panel_sha = api.dataset_info(PANEL_DS).sha
    recent_sha = api.dataset_info(RECENT_DS).sha
    panel_license = (api.dataset_info(PANEL_DS).card_data or {}).get("license")
    recent_license = (api.dataset_info(RECENT_DS).card_data or {}).get("license")

    rng = random.Random(SEED)
    used: set[str] = set()
    candidates: dict[str, list[dict]] = {}

    # ---- glopardo (panel source) -----------------------------------------
    df = load_dataset(PANEL_DS, revision=panel_sha)["train"].to_pandas()
    df = df[select_universe(df)].dropna(subset=["year", "quarter"]).copy()
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df["_tlen"] = df["transcript"].str.len()
    df = df.sort_values("_tlen", ascending=False).drop_duplicates(["ticker", "year", "quarter"])
    df = df[df["_tlen"] >= MIN_CHARS]
    # one candidate call per ticker (seeded) keeps company variety high
    df = df.sample(frac=1.0, random_state=SEED).drop_duplicates("ticker")

    rows = []
    for r in df.itertuples():
        sig = signals(r.transcript)
        rows.append(
            {
                "source": "glopardo",
                "ticker": str(r.ticker),
                "company": str(r.company),
                "sector": str(r.sector),
                "year": int(r.year),
                "quarter": int(r.quarter),
                "text": r.transcript,
                **sig,
            }
        )
    candidates["abstain"] = [r for r in rows if r["tier"] == "abstain"]
    candidates["handoff"] = [r for r in rows if r["tier"] == "handoff"]
    candidates["firstq"] = [r for r in rows if r["tier"] == "firstq"]
    candidates["header_multi"] = [
        r for r in rows if r["tier"] == "header" and r["header_mentions"] >= 2
    ]
    candidates["header_single"] = [
        r for r in rows if r["tier"] == "header" and r["header_mentions"] == 1
    ]
    by_len = sorted((r for r in rows if r["tier"] != "abstain"), key=lambda r: r["n_chars"])
    candidates["extreme"] = by_len[:40] + by_len[-40:]

    # ---- rogersurf (live source) -----------------------------------------
    panel_tickers = set(pd.read_parquet(PANEL_PARQUET, columns=["ticker"])["ticker"])
    raw = load_dataset(RECENT_DS, revision=recent_sha)["train"].to_pandas()
    raw = raw[raw["ticker"].isin(panel_tickers)].copy()
    raw["cd"] = pd.to_datetime(raw["call_date"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["cd"])
    sector_map = (
        pd.read_parquet(PANEL_PARQUET, columns=["ticker", "sector", "company"])
        .drop_duplicates("ticker")
        .set_index("ticker")
    )
    clean = raw["transcript_clean"]
    raw["text"] = clean.where(clean.str.len().fillna(0) >= MIN_CHARS, raw["transcript"])
    raw = raw.dropna(subset=["text"])
    raw = raw[raw["text"].str.len() >= MIN_CHARS]
    raw = raw.sample(frac=1.0, random_state=SEED).drop_duplicates("ticker")

    rs_rows = []
    for r in raw.itertuples():
        sig = signals(r.text)
        rs_rows.append(
            {
                "source": "rogersurf",
                "ticker": str(r.ticker),
                "company": str(sector_map.loc[r.ticker, "company"]),
                "sector": str(sector_map.loc[r.ticker, "sector"]),
                "year": int(r.cd.year),
                "quarter": int((r.cd.month - 1) // 3 + 1),
                "text": r.text,
                **sig,
            }
        )
    candidates["rs_mf"] = [r for r in rs_rows if r["mf_format"] and r["tier"] != "abstain"]
    candidates["rs_colon"] = [r for r in rs_rows if not r["mf_format"] and r["tier"] != "abstain"]
    candidates["rs_abstain"] = [r for r in rs_rows if r["tier"] == "abstain"]

    # ---- bucket quotas ----------------------------------------------------
    quotas = {
        "abstain": 4,
        "header_multi": 6,
        "header_single": 3,
        "firstq": 4,
        "handoff": 4,
        "extreme": 3,
        "rs_mf": 3,
        "rs_colon": 2,
        "rs_abstain": 1,
    }
    selected: list[dict] = []
    for bucket, k in quotas.items():
        got = pick(candidates.get(bucket, []), k, rng, used)
        for g in got:
            g["bucket"] = bucket
        if len(got) < k:
            print(f"WARNING: bucket {bucket} short: {len(got)}/{k}")
        selected.extend(got)

    # ---- split: stratified by bucket, seeded ------------------------------
    rng2 = random.Random(SEED + 1)
    holdout_ids: set[str] = set()
    for bucket in quotas:
        members = [r for r in selected if r["bucket"] == bucket]
        n_hold = max(1, round(len(members) / 3))
        rng2.shuffle(members)
        for m in members[:n_hold]:
            holdout_ids.add(m["ticker"])
    # trim/pad to exactly 10 holdout
    all_tickers = [r["ticker"] for r in selected]
    rng2.shuffle(all_tickers)
    for t in all_tickers:
        if len(holdout_ids) >= 10:
            break
        holdout_ids.add(t)
    while len(holdout_ids) > 10:
        holdout_ids.pop()

    entries = []
    for r in sorted(selected, key=lambda r: (r["bucket"], r["ticker"])):
        entries.append(
            {
                "id": f"{r['source']}-{r['ticker']}-{r['year']}Q{r['quarter']}",
                "source_dataset": PANEL_DS if r["source"] == "glopardo" else RECENT_DS,
                "ticker": r["ticker"],
                "company": r["company"],
                "sector": r["sector"],
                "year": r["year"],
                "quarter": r["quarter"],
                "sha256": sha256(r["text"]),
                "n_chars": r["n_chars"],
                "bucket": r["bucket"],
                "rationale": RATIONALE[r["bucket"]],
                "split": "holdout" if r["ticker"] in holdout_ids else "dev",
            }
        )

    manifest = {
        "description": "Extraction benchmark: 30 real earnings-call transcripts, "
        "selected for difficulty coverage (selection-biased toward hard cases; "
        "see evals/build_manifest.py). Transcripts are NOT committed — fetch "
        "them with `python -m evals.fetch_benchmark`, which verifies sha256.",
        "selection_seed": SEED,
        "sources": {
            PANEL_DS: {"revision": panel_sha, "license_tag": panel_license},
            RECENT_DS: {"revision": recent_sha, "license_tag": recent_license},
        },
        "redistribution_note": "Transcript text is redistributed by the two "
        "public Hugging Face datasets above; this repository does not "
        "re-commit it. Only short boundary-anchor excerpts (<300 chars) "
        "appear in evals/annotations.json.",
        "n_transcripts": len(entries),
        "transcripts": entries,
    }
    OUT.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {len(entries)} transcripts")
    print("buckets:", dict(Counter(e["bucket"] for e in entries)))
    print("sectors:", dict(Counter(e["sector"] for e in entries)))
    print("years:", dict(Counter(e["year"] for e in entries)))
    print("split:", dict(Counter(e["split"] for e in entries)))


RATIONALE = {
    "abstain": "extractor currently abstains: genuinely-missing-Q&A or missed-boundary case",
    "header_multi": "explicit Q&A header with repeated header-phrase mentions (intro announcement trap)",
    "header_single": "explicit Q&A header, single mention (easy anchor case)",
    "firstq": "no usable header; operator 'first question' cue is the boundary evidence",
    "handoff": "neither header nor first-question cue; general analyst-handoff fallback territory",
    "extreme": "length extreme (position-window stress at 15%/95% bounds)",
    "rs_mf": "live source, Motley Fool name/--/title format (missing SA-style headers)",
    "rs_colon": "live source, colon speaker format",
    "rs_abstain": "live source, extractor abstains",
}


if __name__ == "__main__":
    main()
