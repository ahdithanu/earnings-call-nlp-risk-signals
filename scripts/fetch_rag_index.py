"""Download the published retrieval index instead of building it.

The index behind POST /ask is a weekly-refreshed build artifact, published
by the refresh-signals workflow to the rolling `rag-index` GitHub release
(it is deliberately NOT committed: ~36MB that changes weekly would bloat
git history). This script fetches it into data/processed/rag_index/ —
stdlib-only, so it runs in any environment with Python, no install needed.

Usage:  python -m scripts.fetch_rag_index [--out data/processed/rag_index]
Exit codes: 0 fetched; 1 release or asset missing (build locally instead
with `python -m scripts.build_rag_index`).
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "ahdithanu/earnings-call-nlp-risk-signals"
RELEASE_TAG = "rag-index"
FILES = ["meta.json", "chunks.jsonl.gz", "embeddings.npz"]
BASE = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/rag_index")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        url = BASE + name
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
                data = resp.read()
        except urllib.error.HTTPError as exc:
            print(
                f"could not fetch {url} ({exc.code}): the rolling '{RELEASE_TAG}' "
                "release has not been published yet — run the 'Refresh recent "
                "signals' workflow once, or build locally with "
                "`python -m scripts.build_rag_index`"
            )
            return 1
        (out / name).write_bytes(data)
        print(f"fetched {name}: {len(data) / 1e6:.1f} MB")
    print(f"index ready at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
