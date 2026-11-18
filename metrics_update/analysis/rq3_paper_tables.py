"""
LaTeX tables of the RQ3 results, in the format used in the paper.

Usage (from the repository root, after running ``metrics_update/notebooks/RQ3_reading_behavior.ipynb``):

    python metrics_update/analysis/rq3_paper_tables.py [--results-dir DIR] [--output-dir DIR]

Inputs (``--results-dir``, default ``metrics_update/outputs/rq3``):

- ``rq3_all_main_results.csv``: paired contrasts (Smell vs. Clean and perceived types);
- ``rq3_all_predominance_results.csv``: one-sample predominance tests against 0.50;
- ``rq3_predominance_threshold_sensitivity.csv``: Thorough/Skimming predominance by threshold.

Outputs (``--output-dir``, default ``metrics_update/outputs/rq3/paper_tables``), one file per table:

- ``rq3_smell_clean_yuen_phat.tex``           (tab:rq3_smell_clean)
- ``mental_model_bottom_up_predominance.tex`` (tab:mental_model_bottom_up_predominance)
- ``mental_model_significant_contrasts.tex``  (tab:mental_model_contrasts)
- ``reading_direction_predominance.tex``      (tab:reading_direction_predominance)
- ``thorough_predominance.tex``               (tab:thorough_predominance)
- ``thorough_threshold_sensitivity.tex``      (tab:thorough_threshold_sensitivity)
- ``sectionally_predominance.tex``            (tab:sectionally_predominance)
- ``sectionally_contrasts.tex``               (tab:sectionally_contrasts)

"Significant contrasts" tables list the pairwise contrasts with p_FDR < ALPHA. When no contrast of
a family is significant, the file contains only a LaTeX comment saying so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "metrics_update" / "outputs" / "rq3"
OUTPUT_DIR = RESULTS_DIR / "paper_tables"

ALPHA = 0.05
TYPES = ["LM", "DC", "FE", "Others", "None"]

MENTAL_MODEL_STATES = {
    "P(Bottom-Up Inside)": "Bottom-Up Inside",
    "P(Top-Down Inside)": "Top-Down Inside",
    "P(Bottom-Up Outside)": "Bottom-Up Outside",
    "P(Top-Down Outside)": "Top-Down Outside",
}
READING_PATTERNS = {
    "P(Top-to-Bottom)": "Top-to-Bottom",
    "P(Thorough) @ 1000 ms": "Thorough (1,000 ms)",
    "P(Sectionally) @ three-line moving window": "Sectionally (3-line window)",
}
BOTTOM_UP = "P(Bottom-Up overall)"
DIRECTION = "P(Top-to-Bottom)"
THOROUGH = "P(Thorough) @ 1000 ms"
SECTIONALLY = "P(Sectionally) @ three-line moving window"

PREDOMINANCE_LABEL = {"No clear predominance": "Unclear"}
THRESHOLD_MARK = {"Skimming": "S", "Thorough": "T", "No clear predominance": "--"}

PREAMBLE = r"""\begin{{table}}[t]
    \centering
    \footnotesize
    \caption{{{caption}}}
    \label{{{label}}}
    \setlength{{\tabcolsep}}{{3pt}}
    \renewcommand{{\arraystretch}}{{1.15}}
    \resizebox{{\columnwidth}}{{!}}{{%
    \begin{{tabular}}{{{columns}}}
        \toprule
{header}
        \midrule
{body}
        \bottomrule
    \end{{tabular}}%
    }}
\end{{table}}
"""


# --------------------------------------------------------------------------- formatting


def num(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"

    return f"$-${text[1:]}" if text.startswith("-") else text


def ci(low: float, high: float, digits: int = 2) -> str:
    return f"[{num(low, digits)}, {num(high, digits)}]"


def pval(value: float) -> str:
    return "$<0.0001$" if value < 0.0001 else f"{value:.4f}"


def contrast_label(comparison: str) -> str:
    return comparison.replace("vs. ", r"vs.\ ")


def table(caption: str, label: str, columns: str, header: str, rows: list[str], scriptsize=False) -> str:
    text = PREAMBLE.format(
        caption=caption,
        label=label,
        columns=columns,
        header=header,
        body="\n".join(f"        {row} \\\\" for row in rows),
    )

    return text.replace(r"\footnotesize", r"\scriptsize") if scriptsize else text


# --------------------------------------------------------------------------- loading


def read(path: Path) -> pd.DataFrame:
    # "None" is a category name, not a missing value.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def type_predominance(predominance: pd.DataFrame, indicator: str) -> pd.DataFrame:
    rows = predominance.loc[
        predominance["family"].eq("Predominance — perceived types") & predominance["indicator"].eq(indicator)
    ].set_index("condition")

    return rows.loc[TYPES]


def significant_contrasts(main: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    return main.loc[main["family"].isin(families) & main["p_fdr"].lt(ALPHA)]


# --------------------------------------------------------------------------- tables


PREDOMINANCE_HEADER = r"""             &     & \textbf{Trimmed} & \textbf{$\Delta_{0.50}$} & & & \textbf{95\% CI} \\
        \textbf{Type} & \textbf{$n$} & \textbf{mean} & \textbf{(p.p.)} &
        \textbf{$p_{\mathrm{FDR}}$} & \textbf{$\hat{p}_{\mathrm{pred}}$} &
        \textbf{$\hat{p}_{\mathrm{pred}}$} \\"""

CONTRAST_HEADER = r"""        \textbf{Contrast} & \textbf{$n$} & \textbf{$\Delta$ (p.p.)} &
        \textbf{95\% CI $\Delta$} & \textbf{$p_{\mathrm{FDR}}$} &
        \textbf{$\hat{p}$} & \textbf{95\% CI $\hat{p}$} \\"""

DELTA_050 = (
    r"$\Delta_{0.50}$ represents the difference between the participant-level 20\% trimmed mean and 0.50, "
    r"expressed in percentage points."
)


def predominance_rows(rows: pd.DataFrame, with_label: bool = False) -> list[str]:
    out = []
    for condition, row in rows.iterrows():
        cells = [
            f"{condition:<6}",
            str(int(row["n"])),
            f"{row['trimmed_mean']:.3f}",
            num(row["delta_pp"], 2),
            pval(row["p_fdr"]),
            f"{row['p_pred']:.2f}",
            ci(row["p_pred_ci_low"], row["p_pred_ci_high"]),
        ]
        if with_label:
            cells.append(PREDOMINANCE_LABEL.get(row["predominance"], row["predominance"]))
        out.append(" & ".join(cells))

    return out


def contrast_rows(rows: pd.DataFrame) -> list[str]:
    return [
        " & ".join(
            [
                f"{contrast_label(row['comparison']):<15}",
                str(int(row["n"])),
                num(row["delta_pp"], 2),
                ci(row["ci_low_pp"], row["ci_high_pp"]),
                pval(row["p_fdr"]),
                f"{row['phat']:.2f}",
                ci(row["phat_ci_low"], row["phat_ci_high"]),
            ]
        )
        for _, row in rows.iterrows()
    ]


def no_significant_contrast(label: str, families: list[str]) -> str:
    return (
        f"% {label}: no pairwise contrast reached p_FDR < {ALPHA} in\n"
        + "".join(f"%   {family}\n" for family in families)
        + "% (table omitted; see rq3_all_main_results.csv for all contrasts).\n"
    )


def smell_clean_table(main: pd.DataFrame) -> str:
    rows = main.loc[main["family"].eq("Smell vs. Clean")].set_index("indicator")
    n = rows["n"].unique()
    assert len(n) == 1, n

    def block(title: str, indicators: dict[str, str]) -> list[str]:
        lines = [rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}}"]
        for indicator, name in indicators.items():
            row = rows.loc[indicator]
            lines.append(
                " & ".join(
                    [
                        f"{name:<27}",
                        num(row["delta_pp"], 2),
                        ci(row["ci_low_pp"], row["ci_high_pp"]),
                        pval(row["p_value"]),
                        pval(row["p_fdr"]),
                        f"{row['phat']:.2f}",
                        ci(row["phat_ci_low"], row["phat_ci_high"]),
                    ]
                )
            )
        return lines

    body = block("Mental-model state", MENTAL_MODEL_STATES) + block("Reading pattern", READING_PATTERNS)
    body.insert(len(MENTAL_MODEL_STATES) + 1, r"\addlinespace")
    header = r"""        \textbf{Indicator} &
        \textbf{$\Delta$ (p.p.)} &
        \textbf{95\% CI $\Delta$} &
        \textbf{$p$} &
        \textbf{$p_{\mathrm{FDR}}$} &
        \textbf{$\hat{p}$} &
        \textbf{95\% CI $\hat{p}$} \\"""

    text = table(
        caption=(
            r"Paired comparison between snippets perceived as \textit{Smell} and \textit{Clean} for the RQ3 "
            rf"reading-behavior indicators ($n = {int(n[0])}$ participants). Differences ($\Delta$) are defined as "
            r"\textit{Smell} $-$ \textit{Clean} and are reported in percentage points. $p_{\mathrm{FDR}}$ denotes "
            r"the Benjamini--Hochberg-adjusted $p$-value."
        ),
        label="tab:rq3_smell_clean",
        columns="l r c r r r c",
        header=header,
        rows=body,
        scriptsize=True,
    )
    # \addlinespace and \multicolumn rows must not end with "\\" twice; \addlinespace takes none.
    text = text.replace("\\addlinespace \\\\", "\\addlinespace")

    return "% Requires \\usepackage{booktabs} and \\usepackage{graphicx}.\n" + text


def mental_model_contrasts_table(main: pd.DataFrame) -> str:
    families = [f"Mental model — {state}" for state in MENTAL_MODEL_STATES]
    rows = significant_contrasts(main, families)
    if rows.empty:
        return no_significant_contrast("tab:mental_model_contrasts", families)

    states = [MENTAL_MODEL_STATES[i] for i in MENTAL_MODEL_STATES if i in set(rows["indicator"])]
    multiple = len(states) > 1
    if multiple:
        rows = rows.assign(comparison=rows["indicator"].map(MENTAL_MODEL_STATES) + ": " + rows["comparison"])

    return table(
        caption=(
            "FDR-significant pairwise comparisons for the "
            + (" and ".join(states) if multiple else states[0])
            + " mental-model state"
            + ("s" if multiple else "")
            + r" across perceived smell categories. Differences ($\Delta$) are defined as the first category "
            "minus the second and are reported in percentage points."
        ),
        label="tab:mental_model_contrasts",
        columns="l r r c r r c",
        header=CONTRAST_HEADER,
        rows=contrast_rows(rows),
    )


def sectionally_contrasts_table(main: pd.DataFrame) -> str:
    families = ["Reading organization"]
    rows = significant_contrasts(main, families)
    if rows.empty:
        return no_significant_contrast("tab:sectionally_contrasts", families)

    return table(
        caption=(
            "FDR-significant pairwise comparisons for the proportion of Sectionally transitions under the "
            r"primary three-line moving-window operationalization. Differences ($\Delta$) are defined as the "
            "first category minus the second and are reported in percentage points."
        ),
        label="tab:sectionally_contrasts",
        columns="l r r c r r c",
        header=CONTRAST_HEADER,
        rows=contrast_rows(rows),
    )


def bottom_up_predominance_table(predominance: pd.DataFrame) -> str:
    return table(
        caption=(
            "Predominance analysis for the overall Bottom-Up proportion across perceived smell categories. "
            + DELTA_050
        ),
        label="tab:mental_model_bottom_up_predominance",
        columns="l r r r r r c",
        header=PREDOMINANCE_HEADER,
        rows=predominance_rows(type_predominance(predominance, BOTTOM_UP)),
    )


def direction_predominance_table(predominance: pd.DataFrame) -> str:
    return table(
        caption=(
            "Predominance analysis for the proportion of Top-to-Bottom transitions across perceived smell "
            "categories. " + DELTA_050
        ),
        label="tab:reading_direction_predominance",
        columns="l r r r r r c",
        header=PREDOMINANCE_HEADER,
        rows=predominance_rows(type_predominance(predominance, DIRECTION)),
    )


def thorough_predominance_table(predominance: pd.DataFrame) -> str:
    return table(
        caption=(
            "Predominance analysis for the proportion of Thorough fixations across perceived smell categories "
            "using the 1,000-ms operationalization. " + DELTA_050
        ),
        label="tab:thorough_predominance",
        columns="l r r r r r c",
        header=PREDOMINANCE_HEADER,
        rows=predominance_rows(type_predominance(predominance, THOROUGH)),
    )


def sectionally_predominance_table(predominance: pd.DataFrame) -> str:
    header = r"""             &     & \textbf{Trimmed} & \textbf{$\Delta_{0.50}$} & & & & \\
        \textbf{Type} & \textbf{$n$} & \textbf{mean} & \textbf{(p.p.)} &
        \textbf{$p_{\mathrm{FDR}}$} & \textbf{$\hat{p}_{\mathrm{pred}}$} &
        \textbf{95\% CI} & \textbf{Predominance} \\"""

    return table(
        caption=(
            "Predominance analysis for the proportion of Sectionally transitions under the primary three-line "
            r"moving-window operationalization. $\Delta_{0.50}$ is the difference, in percentage points, between "
            r"the participant-level 20\% trimmed mean and 0.50; the 95\% CI refers to $\hat{p}_{\mathrm{pred}}$."
        ),
        label="tab:sectionally_predominance",
        columns="l r r r r r c l",
        header=header,
        rows=predominance_rows(type_predominance(predominance, SECTIONALLY), with_label=True),
    )


def threshold_sensitivity_table(sensitivity: pd.DataFrame) -> str:
    rows = sensitivity.loc[sensitivity["family"].str.contains("perceived types", regex=False)]
    thresholds = sorted(rows["threshold"].unique())

    lines = []
    for condition in TYPES:
        cells = [f"{condition:<6}"]
        for threshold in thresholds:
            row = rows.loc[rows["condition"].eq(condition) & rows["threshold"].eq(threshold)].iloc[0]
            cells.append(f"{row['trimmed_mean']:.3f}\\textsuperscript{{{THRESHOLD_MARK[row['predominance']]}}}")
        lines.append(" & ".join(cells) + r" \\")

    header = " & ".join(["Type", *[f"{t:,} ms" for t in thresholds]]) + r" \\"
    marks = sorted(set(rows["predominance"]))
    legend = "; ".join(
        f"{THRESHOLD_MARK[m]} = {'no evidence of a difference from 0.50' if m == 'No clear predominance' else m}"
        for m in sorted(marks, key=lambda m: THRESHOLD_MARK[m] == "--")
    )
    body = "\n".join(lines)

    return rf"""\begin{{table}}[ht]
\centering
\caption{{Sensitivity of the Thorough/Skimming predominance analysis to the
fixation-duration threshold across perceived code-smell categories. Values
are participant-level 20\% trimmed means of $P_{{Thorough}}$. Superscripts
indicate the predominant pattern after Benjamini--Hochberg correction
within each threshold ({legend}).}}
\label{{tab:thorough_threshold_sensitivity}}
\begin{{tabular}}{{l{"c" * len(thresholds)}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    main_results = read(args.results_dir / "rq3_all_main_results.csv")
    predominance = read(args.results_dir / "rq3_all_predominance_results.csv")
    sensitivity = read(args.results_dir / "rq3_predominance_threshold_sensitivity.csv")

    tables = {
        "rq3_smell_clean_yuen_phat.tex": smell_clean_table(main_results),
        "mental_model_bottom_up_predominance.tex": bottom_up_predominance_table(predominance),
        "mental_model_significant_contrasts.tex": mental_model_contrasts_table(main_results),
        "reading_direction_predominance.tex": direction_predominance_table(predominance),
        "thorough_predominance.tex": thorough_predominance_table(predominance),
        "thorough_threshold_sensitivity.tex": threshold_sensitivity_table(sensitivity),
        "sectionally_predominance.tex": sectionally_predominance_table(predominance),
        "sectionally_contrasts.tex": sectionally_contrasts_table(main_results),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (args.output_dir / name).write_text(text, encoding="utf-8")
        print("Saved:", (args.output_dir / name).resolve().relative_to(ROOT_DIR, walk_up=True))


if __name__ == "__main__":
    main()
