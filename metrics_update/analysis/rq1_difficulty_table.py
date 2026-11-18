"""
RQ1: distribution of perceived code smells by comprehension-difficulty category and self-admitted difficulty.

Usage (from the repository root):

    python metrics_update/analysis/rq1_difficulty_table.py

Inputs
------
- ``data/code_smell_comprehension_data_rq1.xlsx``, sheet ``Code_Smell_Distri_Category_v1``: one row per
  (category, level of comprehension, self-admitted difficulty), listing the citations as
  ``(<participant>, <snippet>, <smells>)`` tuples. Only the category, the difficulty answer and the
  participant/snippet of each citation are used.
- ``metrics_update/data/base_fixations_design_survey.csv``: the analytic base of RQ2/RQ3.

Rules
-----
- Sample: only citations whose ``participant × snippet`` cell is in the analytic base and has a
  questionnaire answer (same participants and snippets as the eye-tracking analyses).
- Perceived smell of each citation: taken from the questionnaire as coded in the base (``perceived_lm``,
  ``perceived_dc``, ``perceived_fe``, ``perceived_other``; ``None`` when the participant answered that the
  snippet has no smell), so the categories are the same as in RQ2/RQ3. A citation counts once in each
  smell the participant reported (multi-label).
- A citation listed twice in the same category and difficulty answer is counted once.
- ``#cited`` = number of citations; ``Total Smells Only`` = DC + LM + FE; ``Total All`` = DC + LM + FE +
  Others + None.

Outputs (``metrics_update/outputs/rq1/``)
-----------------------------------------
- ``table_distribution_code_smells_difficulty.tex``
- ``rq1_difficulty_citations.csv``: one row per citation, with the inclusion decision and both smell labels
  (spreadsheet and questionnaire).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
CODING_PATH = ROOT_DIR / "data" / "code_smell_comprehension_data_rq1.xlsx"
CODING_SHEET = "Code_Smell_Distri_Category_v1"
BASE_PATH = ROOT_DIR / "metrics_update" / "data" / "base_fixations_design_survey.csv"
OUTPUT_DIR = ROOT_DIR / "metrics_update" / "outputs" / "rq1"

SMELLS = {"DC": "Data Class", "LM": "Long Method", "FE": "Feature Envy", "Others": "Others", "None": "None"}
SMELLS_ONLY = ["DC", "LM", "FE"]


def read_citations() -> pd.DataFrame:
    sheet = pd.read_excel(CODING_PATH, sheet_name=CODING_SHEET)
    sheet.columns = ["category", "level", "difficulty", "n_cited", "citations", "smells"]
    sheet["category"] = sheet["category"].ffill().str.replace(r"\s*\(\d+\)\s*$", "", regex=True).str.strip()
    sheet["level"] = sheet["level"].ffill()
    sheet["difficulty"] = sheet["difficulty"].str.strip()

    rows = []
    for _, row in sheet.iterrows():
        tuples = re.findall(r"\(([^)]*)\)", row["citations"])
        assert len(tuples) == row["n_cited"], (row["category"], row["citations"])
        for item in tuples:
            participant, snippet, *labels = (part.strip() for part in item.split(","))
            rows.append(
                {
                    "category": row["category"],
                    "level": row["level"],
                    "difficulty": row["difficulty"],
                    "participant_id": int(participant),
                    "snippet_id": int(snippet),
                    "smells_spreadsheet": ", ".join(labels),
                }
            )

    return pd.DataFrame(rows)


def perceived_smells(base: pd.DataFrame) -> pd.DataFrame:
    cells = base.drop_duplicates(["participant_id", "snippet_id"])
    perceived = pd.DataFrame(
        {
            "participant_id": cells["participant_id"],
            "snippet_id": cells["snippet_id"],
            "DC": cells["perceived_dc"].eq(1),
            "LM": cells["perceived_lm"].eq(1),
            "FE": cells["perceived_fe"].eq(1),
            "Others": cells["perceived_other"].eq(1),
            "None": cells["perceived_smell_yn"].str.strip().str.lower().eq("no"),
        }
    )
    assert perceived[list(SMELLS)].any(axis=1).loc[cells["perceived_smell_yn"].notna().to_numpy()].all()

    return perceived


def count_table(citations: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    table = (
        citations.groupby(["difficulty", "category"])
        .agg(cited=("participant_id", "size"), **{smell: (smell, "sum") for smell in SMELLS})
        .reindex(pd.MultiIndex.from_product([["Yes", "No"], categories], names=["difficulty", "category"]), fill_value=0)
        .astype(int)
    )
    table["smells_only"] = table[SMELLS_ONLY].sum(axis=1)
    table["total_all"] = table[list(SMELLS)].sum(axis=1)

    return table


def to_latex(table: pd.DataFrame) -> str:
    width = max(len(c) for c in table.index.get_level_values("category"))
    columns = ["cited", *SMELLS, "smells_only", "total_all"]

    blocks = []
    for difficulty in ("Yes", "No"):
        part = table.loc[difficulty]
        lines = [
            f"{category:<{width}} & " + " & ".join(f"{int(row[c]):<2}" for c in columns).rstrip() + r" \\"
            for category, row in part.iterrows()
        ]
        total = part[columns].sum()
        lines += [
            r"\midrule",
            r"\textbf{Total} & " + " & ".join(rf"\textbf{{{int(total[c])}}}" for c in columns) + r" \\",
        ]
        blocks.append(
            "\n".join(
                [rf"\multicolumn{{9}}{{l}}{{\textit{{Self-admitted difficulty: {difficulty}}}}} \\", r"\midrule", *lines]
            )
        )

    body = "\n\\midrule\n".join(blocks)

    return rf"""\begin{{table*}}[t]
\footnotesize
\centering
\caption{{Distribution of code smells grouped by comprehension difficulty categories and self-admitted difficulty.}}
\label{{tab:code-smell-categories-Difficulty}}
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabularx}}{{\textwidth}}{{>{{\raggedright\arraybackslash}}X *{{8}}{{c}}}}
\toprule
\textbf{{Category}} & \textbf{{\#cited}} & \textbf{{DC}} & \textbf{{LM}} & \textbf{{FE}} &
\textbf{{Others}} & \textbf{{None}} &
\textbf{{\shortstack{{Total\\Smells Only}}}} & \textbf{{\shortstack{{Total\\All}}}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""


def load_citations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All citations (with inclusion flags) and the citations used in the RQ1 tables."""
    citations = read_citations()
    base = pd.read_csv(BASE_PATH, low_memory=False)

    perceived = perceived_smells(base)
    citations = citations.merge(perceived, on=["participant_id", "snippet_id"], how="left", indicator=True)
    # In the analytic base *and* with a questionnaire answer for that snippet (e.g., participant 1 ×
    # snippet 12 is eye-tracked but has no perception); otherwise the citation would count with no smell.
    citations["in_sample"] = citations.pop("_merge").eq("both") & citations[list(SMELLS)].fillna(False).any(axis=1)
    citations["duplicate"] = citations.duplicated(["category", "difficulty", "participant_id", "snippet_id"])
    citations["smells_questionnaire"] = citations[list(SMELLS)].apply(
        lambda row: ", ".join(SMELLS[s] for s in SMELLS if row[s] is True), axis=1
    )

    used = citations.loc[citations["in_sample"] & ~citations["duplicate"]].copy()
    used[list(SMELLS)] = used[list(SMELLS)].astype(bool)

    return citations, used


def main() -> None:
    citations, used = load_citations()
    categories = list(dict.fromkeys(citations["category"]))
    table = count_table(used, categories)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    citations.to_csv(OUTPUT_DIR / "rq1_difficulty_citations.csv", index=False)
    (OUTPUT_DIR / "table_distribution_code_smells_difficulty.tex").write_text(to_latex(table), encoding="utf-8")

    excluded = citations.loc[~citations["in_sample"]]
    relabeled = used.loc[
        used["smells_spreadsheet"].map(lambda s: set(x.strip() for x in s.split(",")))
        != used["smells_questionnaire"].map(lambda s: set(x.strip() for x in s.split(",")))
    ]
    print(f"Citations in the spreadsheet: {len(citations)}")
    print(f"Excluded (participant × snippet not in the analytic base or without perception): {len(excluded)}")
    print(excluded.groupby(["participant_id", "snippet_id"]).size().rename("citations").to_string())
    print(f"Duplicates removed: {int((citations['in_sample'] & citations['duplicate']).sum())}")
    print(f"Used: {len(used)} citations from {used['participant_id'].nunique()} participants")
    print(f"Smell label differs between spreadsheet and questionnaire: {len(relabeled)}")
    print(relabeled[["category", "difficulty", "participant_id", "snippet_id", "smells_spreadsheet", "smells_questionnaire"]].to_string(index=False))
    print()
    print(table.to_string())
    for name in ("table_distribution_code_smells_difficulty.tex", "rq1_difficulty_citations.csv"):
        print("Saved:", (OUTPUT_DIR / name).relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
