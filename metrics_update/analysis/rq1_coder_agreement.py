"""
RQ1: inter-coder agreement (Cohen's kappa and Gwet's AC1) of the comprehension-difficulty coding.

Usage (from the repository root):

    python metrics_update/analysis/rq1_coder_agreement.py

Input: ``data/code_smell_comprehension_data_rq1.xlsx``, sheets ``Analysis_*``. Each row is a coded
segment (quote) of an answer: column ``Sub categoria`` holds coder 1's category, ``Concordância`` holds
coder 2's review ("Sim" = agrees; "Não - <category>" or a category name = disagrees; empty = segment not
reviewed), ``Experiment ID`` is the participant and the last column is the question (``<snippet>.1``).

Sample: the same rule as ``rq1_difficulty_table.py``: only segments whose ``participant × snippet`` cell is
in the analytic base (``metrics_update/data/base_fixations_design_survey.csv``) with a questionnaire
answer. Kappa is computed on the reviewed segments; the all-segments figure is printed for reference.

Coder 2's review is read under three scenarios that differ only in ambiguous marks:

- lenient: "Sim - <extra code>" = agreement; "Não - creio que seria encapsulamento" = agreement
  (coder 1's label is already Encapsulation, i.e., the consensus label);
- primary: as lenient, except the encapsulation remark counts as disagreement;
- conservative: as primary, and "Sim - <extra code>" counts as disagreement.

Output: ``metrics_update/outputs/rq1/rq1_coder_agreement.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
CODING_PATH = ROOT_DIR / "data" / "code_smell_comprehension_data_rq1.xlsx"
BASE_PATH = ROOT_DIR / "metrics_update" / "data" / "base_fixations_design_survey.csv"
OUTPUT_DIR = ROOT_DIR / "metrics_update" / "outputs" / "rq1"

N_BOOT = 10_000
SEED = 20260101

SYNONYMS = {"Excesso de Variáveis ou Parâmetro": "Excesso de Variáveis ou Parâmetros"}


def normalize(category) -> str:
    text = " ".join(str(category).split())

    return SYNONYMS.get(text, text)


def read_segments() -> pd.DataFrame:
    sheets = pd.read_excel(CODING_PATH, sheet_name=None, header=None)
    rows = []

    for name, sheet in sheets.items():
        if not name.startswith("Analysis_"):
            continue
        body = sheet.iloc[2:]
        for index, row in body.loc[body[2].notna()].iterrows():
            rows.append(
                {
                    "sheet": name,
                    "row": index,
                    "participant_id": int(row[6]),
                    "snippet_id": int(str(row[7]).split(".")[0]),
                    "coder1": normalize(row[2]),
                    "mark": None if pd.isna(row[0]) else " ".join(str(row[0]).split()),
                }
            )

    return pd.DataFrame(rows)


def coder2(row: pd.Series, scenario: str) -> str:
    mark = row["mark"].lower()

    if mark.startswith("sim -"):  # agrees, but also suggests an extra code
        return row["coder1"] if scenario != "conservative" else row["coder1"] + " +extra"
    if mark == "sim":
        return row["coder1"]
    if mark.startswith("não - creio que seria encapsulamento"):
        return row["coder1"] if scenario == "lenient" else "Encapsulamento*"
    if mark.startswith("não - não é pelo"):  # rejects the code without proposing another
        return "(no code)"
    if mark.startswith("não - "):
        return normalize(row["mark"][6:])

    return normalize(row["mark"])  # coder 2 wrote a category name


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    categories = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in categories)

    return (po - pe) / (1 - pe)


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    categories = np.union1d(a, b)
    po = np.mean(a == b)
    pi = np.array([(np.mean(a == c) + np.mean(b == c)) / 2 for c in categories])
    pe = (pi * (1 - pi)).sum() / (len(categories) - 1)

    return (po - pe) / (1 - pe)


def agreement(reviewed: pd.DataFrame, scenario: str, rng: np.random.Generator) -> dict:
    c1 = reviewed["coder1"].to_numpy()
    c2 = reviewed.apply(coder2, axis=1, scenario=scenario).to_numpy()
    kappa = cohen_kappa(c1, c2)
    boot = [cohen_kappa(c1[i], c2[i]) for i in (rng.integers(0, len(c1), len(c1)) for _ in range(N_BOOT))]
    low, high = np.percentile(boot, [2.5, 97.5])

    return {
        "scenario": scenario,
        "n_reviewed": len(c1),
        "n_agree": int((c1 == c2).sum()),
        "observed_agreement": float(np.mean(c1 == c2)),
        "kappa": kappa,
        "kappa_ci_low": low,
        "kappa_ci_high": high,
        "ac1": gwet_ac1(c1, c2),
    }


def main() -> None:
    segments = read_segments()

    base = pd.read_csv(BASE_PATH, usecols=["participant_id", "snippet_id", "perceived_smell_yn"], low_memory=False)
    cells = base.drop_duplicates(["participant_id", "snippet_id"])
    cells = set(map(tuple, cells.loc[cells["perceived_smell_yn"].notna(), ["participant_id", "snippet_id"]].values))
    segments["in_sample"] = [cell in cells for cell in zip(segments["participant_id"], segments["snippet_id"])]

    sample = segments.loc[segments["in_sample"]]
    print(f"Coded segments: {len(segments)} | in the analytic sample: {len(sample)} "
          f"({sample['participant_id'].nunique()} participants)")
    print(f"Reviewed by coder 2: all segments {segments['mark'].notna().sum()} | "
          f"in sample {sample['mark'].notna().sum()} ({sample['mark'].notna().mean():.0%})")

    results = []
    for label, data in (("analytic sample", sample), ("all segments", segments)):
        reviewed = data.loc[data["mark"].notna()]
        for scenario in ("lenient", "primary", "conservative"):
            results.append({"sample": label, **agreement(reviewed, scenario, np.random.default_rng(SEED))})

    results = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "rq1_coder_agreement.csv", index=False)

    print()
    print(results.round(3).to_string(index=False))
    print("Saved:", (OUTPUT_DIR / "rq1_coder_agreement.csv").relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
