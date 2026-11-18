"""
Characteristics of the study code snippets (size, complexity, chunks, MLCQ severity and reviews).

Usage (from the repository root):

    python metrics_update/analysis/snippet_characteristics.py

The MLCQ reviews of the study samples are read from
``metrics_update/data/mlcq_reviews_study_snippets.csv``. To rebuild that file from the full MLCQ
dataset (Madeyski & Lewowski, ``MLCQCodeSmellSamples.csv``), pass ``--mlcq <path>``.

Outputs, in ``metrics_update/outputs/snippets/``:

- ``snippet_characteristics.csv``
- ``snippet_characteristics.tex``
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import lizard
import pandas as pd
from lizard_languages import JavaReader

ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_UPDATE_DIR = ROOT_DIR / "metrics_update"

SNIPPETS_PATH = METRICS_UPDATE_DIR / "data" / "code_snippets.csv"
MLCQ_REVIEWS_PATH = METRICS_UPDATE_DIR / "data" / "mlcq_reviews_study_snippets.csv"
# AOI labels re-derived by relabel_aoi.py (see that script).
FIXATIONS_PATH = METRICS_UPDATE_DIR / "data" / "fixations_aoi_relabeled.csv"
CODE_DIR = ROOT_DIR / "Code_Snippet"
OUTPUT_DIR = METRICS_UPDATE_DIR / "outputs" / "snippets"

SEVERITY_ORDER = ["none", "minor", "major", "critical"]
SEVERITY_RANK = {severity: rank for rank, severity in enumerate(SEVERITY_ORDER)}
SEVERITY_ABBR = {"none": "N", "minor": "Mi", "major": "Ma", "critical": "C"}
SMELL_ABBR = {"long method": "LM", "data class": "DC", "feature envy": "FE"}

START_MARKER = "CODE SNIPPET STARTS HERE"
END_MARKER = "CODE SNIPPET ENDS HERE"


def snippet_file(snippet_id: int) -> Path:
    return next(CODE_DIR.glob(f"*/{snippet_id}_Code_Snippet.java"))


def displayed_region(snippet_id: int) -> tuple[int, int]:
    """
    First and last line of the target code as displayed to participants.

    The displayed files contain the START/END marker comments, so these line numbers are
    offset from the MLCQ start_line/end_line. Fixation line numbers refer to the displayed file.
    """
    lines = snippet_file(snippet_id).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines, 1) if START_MARKER in line)
    end = next(i for i, line in enumerate(lines, 1) if END_MARKER in line)

    return start + 1, end - 1


def is_code_token(token: str) -> bool:
    return not token.isspace() and not token.startswith(("//", "/*"))


def snippet_size(snippet_id: int) -> dict:
    """
    Size of the displayed target code (between the markers) and of the whole displayed file.

    - loc: non-blank lines of the target code;
    - n_tokens: lexical Java tokens of the target code (comments and whitespace excluded);
    - n_chars: non-whitespace characters of the target code;
    - file_loc: non-blank lines of the whole displayed file (markers excluded).
    """
    lines = snippet_file(snippet_id).read_text(encoding="utf-8", errors="replace").splitlines()
    first, last = displayed_region(snippet_id)
    region = lines[first - 1 : last]
    markers = (START_MARKER, END_MARKER)

    return {
        "region_start": first,
        "region_end": last,
        "loc": sum(bool(line.strip()) for line in region),
        "n_tokens": sum(is_code_token(t) for t in JavaReader.generate_tokens("\n".join(region))),
        "n_chars": sum(not c.isspace() for c in "".join(region)),
        "file_loc": sum(bool(line.strip()) and not any(m in line for m in markers) for line in lines),
    }


def snippet_sizes() -> pd.DataFrame:
    snippet_ids = sorted(pd.read_csv(SNIPPETS_PATH)["code_snippet"].astype(int))

    return pd.DataFrame([{"snippet_id": i, **snippet_size(i)} for i in snippet_ids])


def extract_mlcq_reviews(mlcq_path: Path, snippets: pd.DataFrame) -> pd.DataFrame:
    """Keep every MLCQ review of the samples used in the study (all smells)."""
    mlcq = pd.read_csv(mlcq_path, sep=";")
    reviews = mlcq.loc[mlcq["sample_id"].isin(snippets["sample_id"])]
    columns = ["id", "reviewer_id", "sample_id", "smell", "severity", "review_timestamp", "type", "code_name"]

    return reviews[columns].sort_values(["sample_id", "smell", "review_timestamp"]).reset_index(drop=True)


def majority_severity(severities: list[str]) -> str:
    """Most frequent rating; ties are broken toward the more severe level."""
    counts = Counter(severities)
    top = max(counts.values())

    return max((s for s in counts if counts[s] == top), key=SEVERITY_RANK.get)


def lower_median_severity(severities: list[str]) -> str:
    ordered = sorted(severities, key=SEVERITY_RANK.get)

    return ordered[(len(ordered) - 1) // 2]


def snippet_complexity(snippet: pd.Series) -> tuple[int, str, int]:
    """
    McCabe cyclomatic complexity of the marked snippet.

    Method-level snippets: CC of the MLCQ method (helpers shown alongside it are ignored).
    Class-level snippets: WMC, the sum of the CCs of the methods; abstract methods (no body,
    not reported by lizard) count 1.
    """
    path = snippet_file(snippet["code_snippet"])
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first, last = displayed_region(snippet["code_snippet"])
    start, end = first - 1, last + 1

    functions = [
        function
        for function in lizard.analyze_file(str(path)).function_list
        if start < function.start_line < end
    ]

    if snippet["type"] == "function":
        name = snippet["code_name"].split("#")[-1].split(" ")[0].split(".")[-1]
        cc = next(f.cyclomatic_complexity for f in functions if f.name.split("::")[-1] == name)

        return cc, "CC", 1

    n_abstract = sum(
        " abstract " in f" {line} " and line.rstrip().endswith(";")
        for line in lines[start : end - 1]
    )
    wmc = sum(f.cyclomatic_complexity for f in functions) + n_abstract

    return wmc, "WMC", len(functions) + n_abstract


def build_table(snippets: pd.DataFrame, reviews: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    # AOI labels: "before" and "after" are outside the target code; "chunk1".."chunkk" partition it.
    # (chunk_num of "after" is k + 1, so max(chunk_num) would over-count.)
    target_chunks = fixations.loc[fixations["chunk"].str.fullmatch(r"chunk\d+"), ["fixation_target", "chunk"]]
    n_chunks = target_chunks.groupby("fixation_target")["chunk"].nunique()

    rows = []

    for _, snippet in snippets.iterrows():
        snippet_id = int(snippet["code_snippet"])
        complexity, complexity_metric, n_methods = snippet_complexity(snippet)

        same_smell = reviews.loc[
            reviews["sample_id"].eq(snippet["sample_id"])
            & reviews["smell"].eq(snippet["smell"])
        ].sort_values("review_timestamp")
        severities = same_smell["severity"].str.lower().tolist()
        majority = majority_severity(severities)

        rows.append(
            {
                "snippet_id": snippet_id,
                "smell": SMELL_ABBR[snippet["smell"]],
                "granularity": snippet["type"],
                **snippet_size(snippet_id),
                "sloc_mlcq": int(snippet["source_code_lines"]),
                "complexity": complexity,
                "complexity_metric": complexity_metric,
                "n_methods": n_methods,
                "n_chunks": int(n_chunks[snippet_id]),
                "severity_used": snippet["severity"],
                "review_id_used": int(snippet["id"]),
                "n_reviews": len(severities),
                "ratings": "/".join(SEVERITY_ABBR[s] for s in severities),
                "majority_severity": majority,
                "lower_median_severity": lower_median_severity(severities),
                "matches_majority": snippet["severity"] == majority,
            }
        )

    table = pd.DataFrame(rows)
    table["smell"] = pd.Categorical(table["smell"], list(SMELL_ABBR.values()))

    return table.sort_values(["smell", "snippet_id"]).reset_index(drop=True)


def to_latex(table: pd.DataFrame) -> str:
    blocks = []

    for _, group in table.groupby("smell", observed=True):
        blocks.append(
            "\n".join(
                f"{row['snippet_id']} & {row['smell']} & {row['granularity'].capitalize()} & "
                f"{row['loc']} & {row['n_tokens']} & {row['complexity']} & {row['n_chunks']} & "
                f"{row['severity_used'].capitalize()}{'' if row['matches_majority'] else '$^{\\dagger}$'} & "
                f"{row['n_reviews']} & {row['ratings']} & {row['majority_severity'].capitalize()} \\\\"
                for _, row in group.iterrows()
            )
        )

    body = "\n\\midrule\n".join(blocks)

    return rf"""\begin{{table*}}[ht]
\centering
\caption{{Characteristics of the code snippets. LOC = non-blank lines of the target code as displayed to
participants. Tokens = lexical Java tokens of the target code (comments excluded). Complexity = McCabe cyclomatic complexity of the method (CC) for method-level
snippets and Weighted Methods per Class (WMC, sum of method CCs; abstract methods count 1) for class-level
snippets. Chunks = number of areas of interest into which the target code was divided for the eye-tracking analysis.
Reviews = MLCQ reviews of the same sample for the same smell; Ratings lists them in chronological order
(N = none, Mi = minor, Ma = major, C = critical). Aggregation rule: the severity used in the study is the rating
of a single MLCQ review (no aggregation); the majority rating (ties toward the more severe level) is shown for
comparison. $^{{\dagger}}$ Severity used differs from the majority rating.}}
\label{{tab:snippet_characteristics}}
\begin{{tabular}}{{rllrrrrlrll}}
\toprule
ID & Smell & Granularity & LOC & Tokens & Complexity & Chunks & Severity (used) & Reviews & Ratings & Majority \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mlcq",
        type=Path,
        help="full MLCQCodeSmellSamples.csv; rebuilds the local subset of MLCQ reviews",
    )
    args = parser.parse_args()

    snippets = pd.read_csv(SNIPPETS_PATH)

    if args.mlcq is not None:
        extract_mlcq_reviews(args.mlcq, snippets).to_csv(MLCQ_REVIEWS_PATH, index=False)
        print("Saved:", MLCQ_REVIEWS_PATH.relative_to(ROOT_DIR))

    reviews = pd.read_csv(MLCQ_REVIEWS_PATH)
    fixations = pd.read_csv(FIXATIONS_PATH, sep="\t", dtype={"chunk_num": "Int64"})

    table = build_table(snippets, reviews, fixations)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "snippet_characteristics.csv", index=False)
    (OUTPUT_DIR / "snippet_characteristics.tex").write_text(to_latex(table), encoding="utf-8")

    print(table.to_string(index=False))
    print()
    print("Saved:", (OUTPUT_DIR / "snippet_characteristics.csv").relative_to(ROOT_DIR))
    print("Saved:", (OUTPUT_DIR / "snippet_characteristics.tex").relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
