"""Compose results/weekly_brief.md from the insight readouts.

The top of the hedging watchlist, rendered as a readable brief: for each of
the most-elevated companies, the full readout (claim, driver, character,
terms) plus its excerpt receipts. Regenerated alongside the watchlist so
the artifact is always the current week's edition — this file is the
editorial core of a shareable/newsletter product.
"""

import csv
import json
from pathlib import Path

SIGNALS_CSV = Path("results/latest_uncertainty_signals.csv")
INSIGHTS_JSON = Path("data/processed/latest_insights.json")
OUT = Path("results/weekly_brief.md")

TOP_N = 5
ROLE = {"ceo": "CEO", "cfo": "CFO", "exec": "Executive"}


def main() -> None:
    with SIGNALS_CSV.open(encoding="utf-8") as f:
        watchlist = list(csv.DictReader(f))
    with INSIGHTS_JSON.open(encoding="utf-8") as f:
        payload = json.load(f)
    insights = payload["insights"]

    lines = [
        "# Hedging watch — weekly brief",
        "",
        f"_Generated {payload['generated']} from each company's latest call, "
        "scored against its own history. Derived from the data, not written "
        "by hand._",
        "",
    ]
    written = 0
    for row in watchlist:
        ro = insights.get(row["ticker"])
        if ro is None:
            continue
        lines += [
            f"## {written + 1}. {ro['headline']}",
            "",
            f"**{ro['ticker']} · {ro['quarter']}** — {ro['claim']}",
            "",
            f"{ro['driver']} {ro['character']} {ro['terms']}",
            "",
        ]
        for e in ro.get("excerpts", []):
            lines.append(f"> **{ROLE.get(e['speaker_role'], 'Executive')}:** “{e['text']}”")
            lines.append(">")
        if ro.get("excerpts"):
            lines.pop()  # trailing '>'
            lines.append("")
        written += 1
        if written >= TOP_N:
            break

    lines += ["---", "", f"_{next(iter(insights.values()))['caveat']}_", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}: {written} readouts")


if __name__ == "__main__":
    main()
