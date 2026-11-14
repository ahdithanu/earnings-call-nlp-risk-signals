"""Run the extraction benchmark: one command, machine-readable results.

    python -m evals.run_extraction_eval [--split dev|holdout|all]

Reads evals/manifest.json + evals/annotations.json + evals/cache/*.txt
(populate the cache first with `python -m evals.fetch_benchmark`; the
cache is not committed because the transcripts are not ours to
redistribute). Verifies every cached transcript's sha256 and every
annotation's anchor text before scoring anything — a transcript that
fails verification is reported as `invalid` and excluded from accuracy
denominators, never silently skipped.

Writes:
  evals/results/extraction_results.json   full machine-readable results
  evals/results/extraction_failures.md    human-readable failure report

Exit code is 0 even when metrics are poor: annotations are provisional
(model-annotated, unreviewed — see evals/README.md), so these numbers are
INFORMATIONAL and must not gate CI. The scoring-correctness gate
(tests/test_scoring_fixtures.py) is the CI gate; it needs no network.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from evals.metrics import Aggregate, TranscriptResult, evaluate_transcript

EVALS = Path("evals")
RESULTS = EVALS / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--out-prefix", default="extraction")
    args = parser.parse_args()

    manifest = json.loads((EVALS / "manifest.json").read_text(encoding="utf-8"))
    annotations = json.loads((EVALS / "annotations.json").read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in annotations["annotations"]}

    results: list[TranscriptResult] = []
    for entry in manifest["transcripts"]:
        if args.split != "all" and entry["split"] != args.split:
            continue
        rid, split, bucket = entry["id"], entry["split"], entry["bucket"]
        path = EVALS / "cache" / f"{rid}.txt"
        ann = ann_by_id.get(rid)
        if ann is None:
            results.append(TranscriptResult(rid, split, bucket, "invalid", note="no annotation"))
            continue
        if not path.exists():
            results.append(
                TranscriptResult(
                    rid,
                    split,
                    bucket,
                    "invalid",
                    note="transcript not fetched — run `python -m evals.fetch_benchmark`",
                )
            )
            continue
        text = path.read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry["sha256"]:
            results.append(TranscriptResult(rid, split, bucket, "invalid", note="sha256 mismatch"))
            continue
        if ann["qa_start_char"] is not None:
            anchor = ann["anchor_text"]
            at = text[ann["qa_start_char"] : ann["qa_start_char"] + len(anchor)]
            if at != anchor:
                results.append(
                    TranscriptResult(
                        rid,
                        split,
                        bucket,
                        "invalid",
                        note="annotation anchor does not match text at offset",
                    )
                )
                continue
        results.append(evaluate_transcript(text, ann, id=rid, split=split, bucket=bucket))

    agg = Aggregate()
    per_bucket: dict[str, Aggregate] = {}
    invalid = [r for r in results if r.outcome == "invalid"]
    for r in results:
        if r.outcome == "invalid":
            continue
        agg.add(r)
        per_bucket.setdefault(r.bucket, Aggregate()).add(r)

    payload = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "annotation_provenance": annotations["provenance"],
        "annotations_sha256": hashlib.sha256(
            json.dumps(annotations, sort_keys=True).encode()
        ).hexdigest(),
        "tolerance_sentences": 2,
        "n_invalid": len(invalid),
        "invalid": [{"id": r.id, "note": r.note} for r in invalid],
        "aggregate": agg.summary(),
        "per_bucket": {b: a.summary() for b, a in sorted(per_bucket.items())},
        "per_transcript": [asdict(r) for r in results],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"{args.out_prefix}_results_{args.split}.json"
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # ---- human-readable failure report ------------------------------------
    lines = [
        f"# Extraction benchmark failures — split: {args.split}",
        f"generated: {payload['generated']}",
        f"annotations: {annotations['provenance']['origin']} "
        f"(review status: see evals/annotations.json)",
        "",
        f"evaluated: {agg.n} | invalid (excluded): {len(invalid)}",
        f"outcomes: {agg.summary()['outcomes']}",
        "",
    ]
    bad = [
        r
        for r in results
        if r.outcome in ("false_abstain", "false_find")
        or (r.outcome == "found_vs_gold" and not r.within_tolerance)
    ]
    if not bad:
        lines.append("No failures outside tolerance.")
    for r in bad:
        lines.append(
            f"- **{r.id}** [{r.bucket}] {r.outcome}"
            + (
                f": sentence_delta={r.sentence_delta}, char_delta={r.char_delta}, "
                f"contamination={r.contamination:.3f}, recall={r.recall:.3f}"
                if r.outcome == "found_vs_gold"
                else (f" (gold_start={r.gold_start}, predicted={r.predicted})")
            )
            + (" [AMBIGUOUS annotation]" if r.ambiguous else "")
        )
    for r in invalid:
        lines.append(f"- **{r.id}** INVALID: {r.note}")
    out_md = RESULTS / f"{args.out_prefix}_failures_{args.split}.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload["aggregate"], indent=1))
    print(f"\nwrote {out_json}\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
