"""Cross-reference check: are the parsed CEO/CFO names real?

Without an external executive database, the dataset validates itself at the
2018/2019 transcript-format boundary: the roster era (2013-2018) and the
intro-prose era (2019+) name executives through two completely different
parsers, so agreement between them is evidence both are right. For each
ticker, the last 2018 attribution is compared with the first 2019/2020
attribution by surname.

Mismatches are either genuine executive transitions (several CEO changes
cluster around January 2019: ACN, INTU, LRCX, TDY) or parser artifacts;
the report lists them for manual triage rather than scoring them blindly.

Writes results/role_name_continuity.txt.
"""

import io

import pandas as pd
from datasets import load_dataset

from earnings_signals.exec_roles import intro_titles, role_of_title, roster_titles
from earnings_signals.universe import select_universe

OUT_TXT = "results/role_name_continuity.txt"

report = io.StringIO()


def emit(*args) -> None:
    print(*args)
    print(*args, file=report)


def parsed_role_surnames(transcript: str) -> dict[str, str]:
    """{'ceo': surname, 'cfo': surname} as parsed from either header form."""
    pairs = roster_titles(transcript) or intro_titles(transcript)
    out: dict[str, str] = {}
    for name, title in pairs:
        role = role_of_title(title)
        if role in ("ceo", "cfo") and role not in out:
            out[role] = name.split()[-1]
    return out


def main() -> None:
    df = load_dataset("glopardo/sp500-earnings-transcripts")["train"].to_pandas()
    df = df[select_universe(df) & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    rows = []
    for ticker, g in df.groupby("ticker"):
        pre = g[g.year == 2018].sort_values("quarter", ascending=False)
        post = g[g.year.isin([2019, 2020])].sort_values(["year", "quarter"])
        got_pre: dict[str, str] = {}
        got_post: dict[str, str] = {}
        for _, r in pre.iterrows():
            got_pre = parsed_role_surnames(r["transcript"])
            if got_pre:
                break
        for _, r in post.iterrows():
            got_post = parsed_role_surnames(r["transcript"])
            if got_post:
                break
        for role in ("ceo", "cfo"):
            a, b = got_pre.get(role), got_post.get(role)
            if a and b:
                rows.append((ticker, role, a, b, a == b))

    res = pd.DataFrame(rows, columns=["ticker", "role", "pre2019", "post2019", "match"])
    emit("=== role-name continuity across the 2018/2019 format boundary ===")
    emit(f"comparable ticker-roles: {len(res)}")
    emit(f"surname continuity: {res['match'].mean():.1%}")
    emit("\nmismatches (genuine transitions or parser artifacts, triage manually):")
    emit(res[~res.match].to_string(index=False))
    emit(
        "\nknown-genuine CEO transitions in this window: ACN (Nanterme->Rowland), "
        "INTU (Smith->Goodarzi), LRCX (Anstice->Archer), TDY (Mehrabian->Pichelli); "
        "FTNT and STX changed CFOs. ON's 'guttman'/'gutmann' is a transcript "
        "spelling variant of the same person."
    )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
