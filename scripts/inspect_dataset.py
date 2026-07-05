"""Step 1: inspect the glopardo/sp500-earnings-transcripts schema.

Run this BEFORE writing any dataset-specific processing logic. It prints the
split names, column names/dtypes, row count, and one full sample record so the
field mapping (transcript text, speaker structure, sector, EPS, date/quarter
keys) can be decided from what actually exists rather than assumed.

NOTE: requires network access to huggingface.co. In the remote session where
this repo was scaffolded, huggingface.co was blocked by the egress network
policy, so this script could not be run there. Run it locally (or in an
environment where Hugging Face is reachable) and use its output to write the
column mapping in the feature-building pipeline.
"""

import json

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("glopardo/sp500-earnings-transcripts")
    print("=== splits ===")
    print(ds)

    split = list(ds.keys())[0]
    d = ds[split]
    print("\n=== features (columns + dtypes) ===")
    print(d.features)
    print("\n=== num rows ===", d.num_rows)

    print("\n=== one full sample record (long strings truncated) ===")
    row = d[0]
    for key, value in row.items():
        if isinstance(value, str) and len(value) > 2000:
            print(f"\n--- {key} (str, len={len(value)}) first 2000 chars ---")
            print(value[:2000])
        else:
            print(f"\n--- {key} ({type(value).__name__}) ---")
            print(value if isinstance(value, str) else json.dumps(value, default=str)[:2000])


if __name__ == "__main__":
    main()
