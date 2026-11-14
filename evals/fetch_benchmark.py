"""Fetch the benchmark transcripts listed in evals/manifest.json.

Transcripts are not committed to the repository (they are redistributed by
their source datasets, not by us). This script re-downloads the two source
datasets at the manifest's PINNED revisions and materializes each selected
transcript into evals/cache/<id>.txt, verifying its sha256 against the
manifest — retrieval is hash-driven, so it does not depend on any
selection-time ordering details.

Usage: python -m evals.fetch_benchmark        (needs huggingface.co egress)
Exit codes: 0 all fetched+verified; 1 any missing/mismatch (reported).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MANIFEST = Path("evals/manifest.json")
CACHE = Path("evals/cache")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    from datasets import load_dataset

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)

    # candidate texts per (dataset, ticker): every transcript variant the
    # entry could have come from; the sha256 picks the right one
    frames = {}
    for ds_name, meta in manifest["sources"].items():
        frames[ds_name] = load_dataset(ds_name, revision=meta["revision"])["train"].to_pandas()

    failures = []
    fetched = 0
    for entry in manifest["transcripts"]:
        out = CACHE / f"{entry['id']}.txt"
        if out.exists() and sha256(out.read_text(encoding="utf-8")) == entry["sha256"]:
            fetched += 1
            continue
        df = frames[entry["source_dataset"]]
        rows = df[df["ticker"] == entry["ticker"]]
        candidates = []
        if "transcript" in rows.columns:
            candidates += [t for t in rows["transcript"] if isinstance(t, str)]
        if "transcript_clean" in rows.columns:
            candidates += [t for t in rows["transcript_clean"] if isinstance(t, str)]
        match = next((t for t in candidates if sha256(t) == entry["sha256"]), None)
        if match is None:
            failures.append(entry["id"])
            print(f"  MISSING {entry['id']}: no candidate text matches the manifest hash")
            continue
        out.write_text(match, encoding="utf-8")
        fetched += 1
        print(f"  fetched {entry['id']} ({entry['n_chars']} chars)")

    print(f"{fetched}/{len(manifest['transcripts'])} transcripts in {CACHE}/")
    if failures:
        print(f"FAILED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
