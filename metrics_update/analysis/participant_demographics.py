"""
Demographic profile of the study participants (background questionnaire).

Usage (from the repository root):

    python metrics_update/analysis/participant_demographics.py

Source: ``supplementary/answers_experiments.tsv`` (one row per participant, ``experiment_id``).

The analytic sample is the set of participants with both eye-tracking data
(``data/all_exp_fixations_original2_cleaned.csv``) and questionnaire answers:

- participant 7 has fixations but no questionnaire (no perception data);
- participants 2 and 21 answered the questionnaire but have no fixations.

Experience answers are free text. The parser reads the ``<n>Y<m>M`` format; answers it cannot read
are resolved in ``EXPERIENCE_OVERRIDES`` (documented one by one) or left as missing.

Outputs, in ``metrics_update/outputs/participants/``:

- ``participant_demographics.csv`` (one row per participant, coded variables)
- ``participant_demographics_summary.csv`` (counts for the table)
- ``participant_demographics.tex``
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_UPDATE_DIR = ROOT_DIR / "metrics_update"

ANSWERS_PATH = ROOT_DIR / "supplementary" / "answers_experiments.tsv"
FIXATIONS_PATH = ROOT_DIR / "data" / "all_exp_fixations_original2_cleaned.csv"
OUTPUT_DIR = METRICS_UPDATE_DIR / "outputs" / "participants"

STUDY_SMELLS = ["Long Method", "Data Class", "Feature Envy"]

# Free-text answers the Y/M parser cannot read: (participant, column) -> years.
EXPERIENCE_OVERRIDES: dict[tuple[int, str], float | None] = {
    # "21 years of programming and around 6-7 years programming in Java."
    (3, "ProgExperienceYears"): 21.0,
    # "6-7 years." -> midpoint
    (3, "JavaExperienceYears"): 6.5,
    # "more than 10 years" -> lower bound (falls in the top band)
    (6, "JavaExperienceYears"): 10.0,
    # "Desenvolvi em Java por 6 meses."
    (1, "JavaExperienceYears"): 0.5,
    # "Aprendi na graduação, porém não tenho muito conhecimento" (no duration given)
    (8, "JavaExperienceYears"): None,
    # "2021-01-01": read as the start date; years until the questionnaire (15/03/2024)
    (11, "ProgExperienceYears"): round((datetime(2024, 3, 15) - datetime(2021, 1, 1)).days / 365.25, 1),
}

PROG_BANDS = [(0, 1, "$<$ 1 year"), (1, 3, "1--3 years"), (3, 5, "3--5 years"), (5, 10, "5--10 years"), (10, None, "$\\geq$ 10 years")]
JAVA_BANDS = [(0, 1e-9, "None"), (1e-9, 1, "$<$ 1 year"), (1, 5, "1--5 years"), (5, None, "$\\geq$ 5 years")]

EDUCATION = {
    "Secondary school (or lower)": "Secondary school",
    "Bachelor of Science (BSc) / Bachelor of Engineering (BEng)": "Bachelor's degree",
    "Master of Science (MSc) / Master of Engineering (MEng)": "Master's degree",
    "Doctoral (Ph.D.) or higher": "Doctorate",
}

SMELL_FAMILIARITY = {
    "Yes, I know what they are and maybe used some tools": "Knows them (may have used tools)",
    "Somewhat - I heard about them in the past but never dug deeper": "Heard of them, never dug deeper",
    "Not really - I heard about them for the first time in this experiment": "First heard of them in this study",
}



def parse_years(value) -> float | None:
    """Parse '<n>Y<m>M', '<n>Y', '<m>M', '<n> months', '<n>' into years."""
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if re.fullmatch(r"\d+", text):
        return float(text)

    match = re.fullmatch(r"(?:(\d+)\s*y)?\s*(?:(\d+)\s*(?:m|months?))?", text)
    if not match or not any(match.groups()):
        return None

    years, months = (int(g) if g else 0 for g in match.groups())

    return round(years + months / 12, 2)


def experience(row: pd.Series, column: str) -> float | None:
    key = (int(row["participant_id"]), column)
    if key in EXPERIENCE_OVERRIDES:
        return EXPERIENCE_OVERRIDES[key]

    return parse_years(row[column])


def band(value: float | None, bands) -> str:
    if value is None or pd.isna(value):
        return "Not reported"

    for low, high, label in bands:
        if value >= low and (high is None or value < high):
            return label

    raise ValueError(value)


def occupation(current_role: str) -> str:
    roles = [r.strip().lower() for r in current_role.split(",")]
    is_student = any(r.startswith("student") for r in roles)
    # Every non-student option of CurrentRole is a professional role (developer, data scientist,
    # architect, lecturer, project manager, "other management role", ...).
    works = any(r and not r.startswith("student") for r in roles)

    if is_student and works:
        return "Student and practitioner"
    if is_student:
        return "Student only"

    return "Practitioner (not a student)"


def student_level(current_role: str) -> str | None:
    for level, label in (("phd", "PhD"), ("master", "Master's"), ("bachelor", "Bachelor's")):
        if f"student - {level} level" in current_role.lower():
            return label

    return None


def code_participants(answers: pd.DataFrame) -> pd.DataFrame:
    data = answers.rename(columns={"experiment_id": "participant_id"}).copy()
    data["participant_id"] = data["participant_id"].astype(int)

    recognizable = data["RecognizableSmells"].fillna("")

    coded = pd.DataFrame(
        {
            "participant_id": data["participant_id"],
            "occupation": data["CurrentRole"].map(occupation),
            "student_level": data["CurrentRole"].map(student_level),
            "education": data["HighestEducation"].map(EDUCATION),
            "prog_years": data.apply(experience, axis=1, column="ProgExperienceYears"),
            "java_years": data.apply(experience, axis=1, column="JavaExperienceYears"),
            "smell_familiarity": data["KnowsCodeSmells"].map(SMELL_FAMILIARITY),
            **{
                f"recognizes_{smell.lower().replace(' ', '_')}": recognizable.str.contains(smell, regex=False)
                for smell in STUDY_SMELLS
            },
            "current_role_raw": data["CurrentRole"],
            "prog_raw": data["ProgExperienceYears"],
            "java_raw": data["JavaExperienceYears"],
        }
    )
    coded["recognizes_all_study_smells"] = coded[[f"recognizes_{s.lower().replace(' ', '_')}" for s in STUDY_SMELLS]].all(axis=1)
    coded["prog_band"] = coded["prog_years"].map(lambda v: band(v, PROG_BANDS))
    coded["java_band"] = coded["java_years"].map(lambda v: band(v, JAVA_BANDS))

    assert coded[["occupation", "education", "smell_familiarity"]].notna().all().all()

    return coded


def summarize(coded: pd.DataFrame) -> pd.DataFrame:
    n = len(coded)
    rows = []

    def add(section: str, label: str, count: int):
        rows.append({"section": section, "category": label, "n": int(count), "pct": 100 * count / n})

    def add_counts(section: str, column: str, order: list[str]):
        counts = coded[column].value_counts()
        unexpected = set(counts.index) - set(order)
        assert not unexpected, unexpected
        for label in order:
            if counts.get(label, 0):
                add(section, label, counts[label])

    add_counts("Occupation", "occupation", ["Student only", "Student and practitioner", "Practitioner (not a student)"])
    add_counts("Student level", "student_level", ["Bachelor's", "Master's", "PhD"])
    add_counts("Highest completed education", "education", list(EDUCATION.values()))
    add_counts("Programming experience", "prog_band", [b[2] for b in PROG_BANDS] + ["Not reported"])
    add_counts("Java experience", "java_band", [b[2] for b in JAVA_BANDS] + ["Not reported"])
    add_counts("Familiarity with code smells", "smell_familiarity", list(SMELL_FAMILIARITY.values()))

    for smell in STUDY_SMELLS:
        add("Self-reported ability to recognize", smell, coded[f"recognizes_{smell.lower().replace(' ', '_')}"].sum())
    add("Self-reported ability to recognize", "All three", coded["recognizes_all_study_smells"].sum())

    return pd.DataFrame(rows)


def median_iqr(values: pd.Series) -> str:
    values = values.dropna()
    q1, med, q3 = values.quantile([0.25, 0.5, 0.75])

    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def to_latex(summary: pd.DataFrame, coded: pd.DataFrame) -> str:
    n = len(coded)
    medians = {
        "Programming experience": median_iqr(coded["prog_years"]),
        "Java experience": median_iqr(coded["java_years"]),
    }

    blocks = []
    for section, group in summary.groupby("section", sort=False):
        header = rf"\multicolumn{{3}}{{l}}{{\textit{{{section}}}}} \\"
        if section in medians:
            header = rf"\multicolumn{{3}}{{l}}{{\textit{{{section}}} (median [IQR] = {medians[section]} years)}} \\"
        lines = [
            rf"\quad {row.category} & {row.n} & {row.pct:.0f}\% \\" for row in group.itertuples()
        ]
        blocks.append("\n".join([header, *lines]))

    body = "\n\\midrule\n".join(blocks)

    return rf"""\begin{{table}}[ht]
\centering
\caption{{Demographic profile of the participants ($n = {n}$). Occupation is the self-reported current role:
students who also reported a professional role (e.g., junior developer, data scientist) are counted as students and
practitioners. Experience bands are lower-inclusive. Recognition of the study smells is the self-reported list of
smells the participant can recognize; categories are not mutually exclusive.}}
\label{{tab:demographics}}
\begin{{tabular}}{{lrr}}
\toprule
Characteristic & $n$ & \% \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    answers = pd.read_csv(ANSWERS_PATH, sep="\t", dtype=str)
    eye_tracked = set(pd.read_csv(FIXATIONS_PATH, sep="\t", usecols=["experiment"])["experiment"].astype(int))

    coded = code_participants(answers)
    coded["in_analytic_sample"] = coded["participant_id"].isin(eye_tracked)
    sample = coded.loc[coded["in_analytic_sample"]].reset_index(drop=True)

    print(f"Questionnaire: {len(coded)} | eye-tracked: {len(eye_tracked)} | analytic sample: {len(sample)}")
    print("Questionnaire without fixations:", sorted(set(coded["participant_id"]) - eye_tracked))
    print("Fixations without questionnaire:", sorted(eye_tracked - set(coded["participant_id"])))

    summary = summarize(sample)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coded.to_csv(OUTPUT_DIR / "participant_demographics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "participant_demographics_summary.csv", index=False)
    (OUTPUT_DIR / "participant_demographics.tex").write_text(to_latex(summary, sample), encoding="utf-8")

    print()
    print(summary.to_string(index=False, float_format="%.0f"))
    print()
    for name in ("participant_demographics.csv", "participant_demographics_summary.csv", "participant_demographics.tex"):
        print("Saved:", (OUTPUT_DIR / name).relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
