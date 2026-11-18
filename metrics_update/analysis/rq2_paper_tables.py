"""Generate the RQ2 LaTeX tables of the paper from the RQ2 notebook outputs.

Run after ``notebooks/RQ2_cognitive_effort.ipynb``:
    .venv/bin/python metrics_update/analysis/rq2_paper_tables.py

Outputs (``metrics_update/outputs/rq2/paper_tables``):
- smell_clean_yuen_phat.tex      (tab:smell_clean_yuen_phat)
- perceived_smell_types_tft.tex  (tab:perceived_smell_types_tft)
- mlcq_agreement_fixation.tex    (tab:mlcq_agreement_fixation)
- rq2_size_normalized.tex        (tab:rq2_size_normalized)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RQ2 = ROOT / "outputs" / "rq2"
OUT = RQ2 / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05


def num(v: float, d: int = 2) -> str:
    s = f"{v:,.{d}f}"
    return s.replace("-", "$-$")


def ci(lo: float, hi: float, d: int = 2) -> str:
    return f"[{num(lo, d)}, {num(hi, d)}]"


def pv(p: float) -> str:
    return "$<0.0001$" if p < 0.0001 else f"{p:.4f}"


def bold_row(cells: list[str], bold: bool) -> str:
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return "        " + " & ".join(cells) + " \\\\"


def smell_clean() -> None:
    d = pd.read_csv(RQ2 / "raw_smell_clean_results.csv")
    rows = []
    for _, r in d.iterrows():
        cells = [r["metric"], str(int(r["n"])), num(r["delta"]), ci(r["ci_low"], r["ci_high"]),
                 f"{num(r['t'])} ({int(r['df'])})", pv(r["p_value"]), f"{r['phat']:.2f}",
                 ci(r["phat_ci_low"], r["phat_ci_high"])]
        rows.append(bold_row(cells, r["metric"].startswith("TFT")))
    tex = r"""\begin{table*}[!tp]
    \centering
    \footnotesize
    \caption{Paired comparison between snippets perceived as Smell and Clean using Yuen's test and the probability of superiority (fixations $\leq$ 5,000~ms). $\Delta$ is defined as \textit{Smell} $-$ \textit{Clean}; therefore, positive values indicate higher values in the \textit{Smell} condition. TFT (bold) is the primary outcome.}
    \label{tab:smell_clean_yuen_phat}
    \setlength{\tabcolsep}{5pt}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{l c r c c c c c}
        \toprule
        \textbf{Metric} & \textbf{$n$} & \textbf{$\Delta$ Trimmed mean} & \textbf{95\% CI $\Delta$} &
        \textbf{$t(df)$} & \textbf{$p$} & \textbf{$\hat{p}$} & \textbf{95\% CI $\hat{p}$} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}
\end{table*}
"""
    (OUT / "smell_clean_yuen_phat.tex").write_text(tex, encoding="utf-8")


def perceived_types() -> None:
    d = pd.read_csv(RQ2 / "raw_perceived_type_results.csv")
    rows = []
    for _, r in d.iterrows():
        cells = [r["comparison"], str(int(r["n"])), num(r["delta"]), ci(r["ci_low"], r["ci_high"]),
                 pv(r["p_value"]), pv(r["p_fdr"]), f"{r['phat']:.2f}", ci(r["phat_ci_low"], r["phat_ci_high"])]
        rows.append(bold_row(cells, r["p_fdr"] < ALPHA))
    tex = r"""\begin{table*}[!tp]
    \centering
    \footnotesize
    \caption{Pairwise comparisons of TFT across code smell categories perceived by participants (fixations $\leq$ 5,000~ms).
    $\Delta$ is defined as Group A $-$ Group B; positive values indicate higher TFT for Group A.
    $p_{\mathrm{FDR}}$ denotes the $p$-value adjusted using the Benjamini--Hochberg false discovery rate procedure; rows with $p_{\mathrm{FDR}} < 0.05$ are in bold.}
    \label{tab:perceived_smell_types_tft}
    \setlength{\tabcolsep}{4pt}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{l c r c c c c c}
        \toprule
        \textbf{Comparison} & \textbf{$n$} & \textbf{$\Delta$ Trimmed mean} & \textbf{95\% CI $\Delta$} &
        \textbf{$p$} & \textbf{$p_{\mathrm{FDR}}$} & \textbf{$\hat{p}$} & \textbf{95\% CI $\hat{p}$} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}
\end{table*}
"""
    (OUT / "perceived_smell_types_tft.tex").write_text(tex, encoding="utf-8")


def mlcq() -> None:
    d = pd.read_csv(RQ2 / "mlcq_agreement_tft_yuen_phat.csv")
    rows = []
    for _, r in d.iterrows():
        cells = [r["metric"], str(int(r["n"])), num(r["delta"]), ci(r["ci_low"], r["ci_high"]),
                 f"{num(r['t'])} ({int(r['df'])})", pv(r["p_value"]), f"{r['phat']:.2f}",
                 ci(r["phat_ci_low"], r["phat_ci_high"])]
        rows.append(bold_row(cells, False))
    tex = r"""\begin{table}[!htbp]
    \centering
    \footnotesize
    \caption{Paired comparisons of fixation metrics between MLCQ-consistent
    and MLCQ-inconsistent classifications (fixations $\leq$ 5,000~ms). Differences ($\Delta$) are
    defined as MLCQ-consistent minus MLCQ-inconsistent; therefore, negative
    values indicate higher values for MLCQ-inconsistent classifications.}
    \label{tab:mlcq_agreement_fixation}
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{1.15}
    \resizebox{\columnwidth}{!}{%
    \begin{tabular}{l r r c r r r c}
        \toprule
        \textbf{Metric} & \textbf{$n$} & \textbf{$\Delta$} &
        \textbf{95\% CI $\Delta$} & \textbf{$t$ (df)} & \textbf{$p$} &
        \textbf{$\hat{p}$} & \textbf{95\% CI $\hat{p}$} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}%
    }
\end{table}
"""
    (OUT / "mlcq_agreement_fixation.tex").write_text(tex, encoding="utf-8")


def size_normalized() -> None:
    sc = pd.read_csv(RQ2 / "rq2_size_normalized_smell_clean.csv")
    sc = sc[sc["family"].str.startswith("Size-normalized")]
    ty = pd.read_csv(RQ2 / "rq2_size_normalized_types.csv")

    def row(label: str, r: pd.Series, d: int = 0) -> str:
        return (f"{label} & {int(r['n'])} & {num(r['delta'], d)} & {ci(r['ci_low'], r['ci_high'], d)} & "
                f"{r['p_fdr']:.2f} & {r['phat']:.2f} {ci(r['phat_ci_low'], r['phat_ci_high'])} \\\\")

    lines = [r"\multicolumn{6}{@{}l}{\textit{Smell vs. Clean}} \\"]
    labels = {"TFT per line (ms/line)": "TFT/line (ms)", "TFT per token (ms/token)": "TFT/token (ms)",
              "FC per line (fixations/line)": "FC/line"}
    for _, r in sc.iterrows():
        lab = labels.get(r["metric"], r["metric"])
        lines.append(row(lab, r, 2 if lab == "FC/line" else 0))
    for metric, title in [("TFT per line (ms/line)", "TFT per line (ms/line)"),
                          ("TFT per token (ms/token)", "TFT per token (ms/token)")]:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{Perceived types --- {title}}}}} \\")
        for _, r in ty[ty["metric"] == metric].iterrows():
            lines.append(row(r["comparison"], r))
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Sensitivity of the RQ2 results to snippet size, using fixations ($\leq$ 5,000~ms) inside the target snippet divided by its lines of code or tokens. $\Delta$: difference in 20\% trimmed means (first minus second). $p_{\mathrm{FDR}}$: Benjamini--Hochberg-adjusted within each block. $\hat{p}$: paired probability of superiority.}
\label{tab:rq2_size_normalized}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrl@{}}
\toprule
Comparison & $n$ & $\Delta$ & 95\% CI $\Delta$ & $p_{\mathrm{FDR}}$ & $\hat{p}$ [95\% CI] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "rq2_size_normalized.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    smell_clean()
    perceived_types()
    mlcq()
    size_normalized()
    for f in sorted(OUT.glob("*.tex")):
        print("Saved:", f.relative_to(ROOT.parent))
