"""
RQ1: frequency of perceived code smells by level of comprehension and self-admitted difficulty.

Usage (from the repository root):

    python metrics_update/analysis/rq1_level_table.py

Uses the same citations as ``rq1_difficulty_table.py`` (same sample, questionnaire smell labels, no
duplicates), so the column totals match the totals of the difficulty-category table. Each citation counts
once in each smell the participant reported; the level of comprehension and the difficulty answer are those
recorded with the citation in the coding spreadsheet.

Output: ``metrics_update/outputs/rq1/table_frequency_difficulty.tex``
"""

from __future__ import annotations

import pandas as pd

from rq1_difficulty_table import OUTPUT_DIR, ROOT_DIR, SMELLS, count_table, load_citations

LEVELS = {
    "Not Understand": "No understanding",
    "Basic": "Basic",
    "Intermediate": "Intermediate",
    "Full": "Full",
}
COLUMNS = ["None", "DC", "LM", "FE", "Others"]


def level_table(used: pd.DataFrame) -> pd.DataFrame:
    assert set(used["level"]) <= set(LEVELS), set(used["level"]) - set(LEVELS)

    counts = used.groupby(["level", "difficulty"])[COLUMNS].sum().unstack("difficulty", fill_value=0)
    counts = counts.reindex(index=list(LEVELS), fill_value=0)
    counts = counts.reindex(columns=pd.MultiIndex.from_product([COLUMNS, ["Yes", "No"]]), fill_value=0).astype(int)
    counts.index = counts.index.map(LEVELS)

    return counts


def with_totals(counts: pd.DataFrame) -> pd.DataFrame:
    """Append the row totals (Y, N, % Y) and the column totals."""
    table = counts.copy()
    table[("Total", "Yes")] = counts.xs("Yes", axis=1, level=1).sum(axis=1)
    table[("Total", "No")] = counts.xs("No", axis=1, level=1).sum(axis=1)
    table.loc["Total"] = table.sum()
    table[("Total", "% Yes")] = (
        100 * table[("Total", "Yes")] / (table[("Total", "Yes")] + table[("Total", "No")])
    ).round().astype(int)

    return table


def to_latex(counts: pd.DataFrame) -> str:
    table = with_totals(counts)
    width = max(len(i) for i in table.index.drop("Total"))

    lines = [
        f"{level:<{width}} & " + " & ".join(f"{v:<2}" for v in row).rstrip() + r" \\"
        for level, row in table.drop(index="Total").iterrows()
    ]
    total = r"\textbf{Total}".ljust(width) + " & " + " & ".join(rf"\textbf{{{v}}}" for v in table.loc["Total"]) + r" \\"
    body = "\n".join(lines)

    return rf"""% Requires \usepackage{{booktabs}}.
\begin{{table*}}[t]
\footnotesize
\centering
\caption{{Frequency of perceived code smells grouped by level of comprehension and self-admitted difficulty (Y = yes, N = no). Counts are the citations of Table~\ref{{tab:code-smell-categories-Difficulty}}, each counted once per perceived smell.}}
\label{{tab:smell-frequency-level-difficulty}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}l *{{10}}{{c}} @{{\hspace{{10pt}}}} ccc@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{\textbf{{None}}}} & \multicolumn{{2}}{{c}}{{\textbf{{DC}}}}
& \multicolumn{{2}}{{c}}{{\textbf{{LM}}}}   & \multicolumn{{2}}{{c}}{{\textbf{{FE}}}}
& \multicolumn{{2}}{{c}}{{\textbf{{Others}}}} & \multicolumn{{3}}{{c}}{{\textbf{{Total}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}\cmidrule(lr){{10-11}}\cmidrule(l){{12-14}}
\textbf{{Level of comprehension}} & Y & N & Y & N & Y & N & Y & N & Y & N & Y & N & \% Y \\
\midrule
{body}
\midrule
{total}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main() -> None:
    citations, used = load_citations()
    counts = level_table(used)

    # Consistency with the difficulty-category table: same column totals.
    by_category = count_table(used, list(dict.fromkeys(citations["category"]))).groupby(level="difficulty").sum()
    for smell in COLUMNS:
        for difficulty in ("Yes", "No"):
            assert counts[(smell, difficulty)].sum() == by_category.loc[difficulty, smell], (smell, difficulty)

    output = OUTPUT_DIR / "table_frequency_difficulty.tex"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(to_latex(counts), encoding="utf-8")

    print(f"Citations used: {len(used)} from {used['participant_id'].nunique()} participants")
    print(with_totals(counts).to_string())
    print("Saved:", output.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
