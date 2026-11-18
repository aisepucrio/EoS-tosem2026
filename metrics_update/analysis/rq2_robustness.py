"""RQ2 robustness analyses requested in the pre-submission review.

1. Linear mixed models of log TFT with crossed random intercepts for
   participant and snippet (and, for comparison, the participant-only model
   previously reported), with and without log snippet size.
2. Fixation-duration quality checks (distribution, extreme fixations).
3. Sensitivity of the primary Smell-Clean and LM-None Yuen tests, and of the
   crossed mixed model, to excluding implausibly long fixations.

Run from the repository root:
    .venv/bin/python metrics_update/analysis/rq2_robustness.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from fixation_filter import MAX_FIXATION_MS  # noqa: E402
from robust_stats import paired_phat, paired_phat_bootstrap_ci, yuen_paired  # noqa: E402

DATA = ROOT / "data" / "base_fixations_design_survey.csv"
SIZES = ROOT / "outputs" / "rq2" / "rq2_snippet_sizes.csv"
OUT = ROOT / "outputs" / "rq2_robustness"
OUT.mkdir(parents=True, exist_ok=True)

TRIM, ALPHA, N_BOOT, SEED = 0.20, 0.05, 10_000, 20260101
KEY = ["participant_id", "snippet_id"]
TYPES = {"lm": "perceived_lm", "dc": "perceived_dc", "fe": "perceived_fe", "other": "perceived_other"}
CUTOFFS = [None, 5000, 2000]  # exclude fixations longer than this (ms)


def snippet_level(fix: pd.DataFrame) -> pd.DataFrame:
    cols = ["perceived_smell_yn", *TYPES.values()]
    agg = fix.groupby(KEY).agg(
        fc=("duration", "size"),
        afd=("duration", "mean"),
        tft=("duration", "sum"),
        **{c: (c, "first") for c in cols},
    )
    inside = fix[fix["inside"]].groupby(KEY)["duration"].sum().rename("tft_inside")
    d = agg.join(inside).reset_index()
    d["tft_inside"] = d["tft_inside"].fillna(0)
    yn = d["perceived_smell_yn"].astype(str).str.strip().str.lower()
    d["perceived_group"] = yn.map({"yes": "Smell", "no": "Clean"})
    d = d.dropna(subset=["perceived_group"]).copy()
    for c in TYPES.values():
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["smell"] = d["perceived_group"].eq("Smell").astype(int)
    sizes = pd.read_csv(SIZES)[["snippet_id", "loc", "n_tokens"]]
    d = d.merge(sizes, on="snippet_id", how="left")
    d["log_tokens"] = np.log(d["n_tokens"])
    d["log_loc"] = np.log(d["loc"])
    return d


def fit_lmm(d: pd.DataFrame, outcome: str, size: str | None, rhs: str, crossed: bool) -> list[dict]:
    d = d.copy()
    d["y"] = np.log(d[outcome].clip(lower=1))
    formula = "y ~ " + (f"{size} + " if size else "") + rhs
    if crossed:
        d["one"] = 1
        model = smf.mixedlm(
            formula, d, groups=d["one"], re_formula="0",
            vc_formula={"participant": "0+C(participant_id)", "snippet": "0+C(snippet_id)"},
        )
    else:
        model = smf.mixedlm(formula, d, groups=d["participant_id"])
    res = model.fit(reml=True)
    ci = res.conf_int()
    rows = []
    for term in res.params.index:
        if term == "Intercept" or "Var" in term:
            continue
        rows.append({
            "outcome": outcome,
            "random_effects": "participant + snippet" if crossed else "participant",
            "size_covariate": size or "none",
            "predictors": "smell" if rhs == "smell" else "types",
            "term": term,
            "b": res.params[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "p": res.pvalues[term],
            "ratio_exp_b": float(np.exp(res.params[term])),
            "snippet_var": res.params.get("snippet Var", np.nan),
            "participant_var": res.params.get("participant Var", res.params.get("Group Var", np.nan)),
            "n_obs": int(res.nobs),
        })
    return rows


def paired_test(d: pd.DataFrame, metric: str, a_mask: pd.Series, b_mask: pd.Series, label: str) -> dict:
    a = d[a_mask].groupby("participant_id")[metric].mean()
    b = d[b_mask].groupby("participant_id")[metric].mean()
    pairs = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    y = yuen_paired(pairs["a"], pairs["b"], trim=TRIM, alpha=ALPHA)
    ph = paired_phat(pairs["a"], pairs["b"])
    bt = paired_phat_bootstrap_ci(pairs["a"], pairs["b"], n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
    return {
        "comparison": label, "metric": metric, "n": y["n"],
        "delta": y["trimmed_mean_diff"], "ci_low": y["ci_low"], "ci_high": y["ci_high"],
        "p": y["p_value"], "phat": ph["phat"],
        "phat_ci_low": bt["phat_ci_low"], "phat_ci_high": bt["phat_ci_high"],
    }


def main() -> None:
    fix = pd.read_csv(DATA)

    # ---------------- 1. fixation-duration quality ----------------
    dur = fix["duration"]
    q = dur.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    quality = {
        "n_fixations": len(fix),
        "n_participants": fix["participant_id"].nunique(),
        "min_ms": dur.min(), "max_ms": dur.max(), "mean_ms": dur.mean(),
        **{f"q{int(k * 100)}_ms": v for k, v in q.items()},
    }
    for t in [1000, 2000, 5000, 10000, 60000]:
        long = fix[dur > t]
        quality[f"n_gt_{t}"] = len(long)
        quality[f"share_fix_gt_{t}"] = len(long) / len(fix)
        quality[f"share_time_gt_{t}"] = long["duration"].sum() / dur.sum()
        quality[f"participants_gt_{t}"] = long["participant_id"].nunique()
    pd.Series(quality).to_csv(OUT / "fixation_duration_quality.csv", header=["value"])
    fix.nlargest(15, "duration")[KEY + ["duration", "inside", "source_file_line"]].to_csv(
        OUT / "longest_fixations.csv", index=False
    )

    # ---------------- 2 & 3. models and sensitivity ----------------
    lmm_rows, yuen_rows = [], []
    rhs_types = " + ".join(TYPES.values())
    for cut in CUTOFFS:
        sub = fix if cut is None else fix[fix["duration"] <= cut]
        d = snippet_level(sub)
        tag = "all fixations" if cut is None else f"fixations <= {cut} ms"

        for outcome in ["tft", "tft_inside"]:
            for crossed in [False, True]:
                for size in [None, "log_tokens", "log_loc"]:
                    for rhs in ["smell", rhs_types]:
                        for r in fit_lmm(d, outcome, size, rhs, crossed):
                            r["fixation_filter"] = tag
                            lmm_rows.append(r)

        none_mask = d["perceived_group"].eq("Clean")
        for metric in ["tft", "afd", "fc"]:
            r = paired_test(d, metric, d["perceived_group"].eq("Smell"), none_mask, "Smell vs. Clean")
            r["fixation_filter"] = tag
            yuen_rows.append(r)
        r = paired_test(d, "tft", d["perceived_lm"].eq(1), none_mask, "LM vs. None")
        r["fixation_filter"] = tag
        yuen_rows.append(r)

    lmm = pd.DataFrame(lmm_rows)
    yuen = pd.DataFrame(yuen_rows)
    lmm.to_csv(OUT / "rq2_lmm_all_specifications.csv", index=False)
    yuen.to_csv(OUT / "rq2_long_fixation_sensitivity_yuen.csv", index=False)

    # ---------------- LaTeX table for the paper ----------------
    write_latex_table(lmm)

    pd.set_option("display.width", 220)
    print(pd.Series(quality).round(3).to_string())
    print(yuen.round(4).to_string())
    key = lmm[(lmm["outcome"] == "tft") & (lmm["fixation_filter"] == f"fixations <= {MAX_FIXATION_MS} ms")]
    print(key[["random_effects", "size_covariate", "predictors", "term", "b", "p"]].round(3).to_string())


def write_latex_table(lmm: pd.DataFrame) -> None:
    sel = lmm[
        (lmm["outcome"] == "tft")
        & (lmm["fixation_filter"] == f"fixations <= {MAX_FIXATION_MS} ms")
        & (lmm["size_covariate"].isin(["none", "log_tokens"]))
    ]
    labels = {
        "smell": "Smell (vs.\\ Clean)", "perceived_lm": "LM",
        "perceived_dc": "DC", "perceived_fe": "FE",
        "perceived_other": "Others", "log_tokens": "log(tokens)",
    }
    order = ["smell", "perceived_lm", "perceived_dc", "perceived_fe", "perceived_other", "log_tokens"]
    specs = [
        ("participant", "log_tokens", "A"),
        ("participant + snippet", "none", "B"),
        ("participant + snippet", "log_tokens", "C"),
    ]

    def fmt(v: float) -> str:
        return f"{v:.2f}".replace("-", "$-$")

    def cell(re, size, term):
        rows = sel[(sel["random_effects"] == re) & (sel["size_covariate"] == size) & (sel["term"] == term)]
        if term == "log_tokens":
            rows = rows[rows["predictors"] == "types"]
        if rows.empty:
            return "--"
        r = rows.iloc[0]
        p = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        return f"{fmt(r['b'])} ({p})"

    lines = [
        "\\begin{table}[!htbp]",
        "\\caption{Linear mixed models of log-transformed TFT (fixations $\\leq$ 5,000~ms). Cells show the "
        "coefficient $b$ and its $p$-value; $e^{b}$ is the multiplicative change in TFT. Random intercepts: "
        "A, participant; B and C, participant and snippet (crossed). Size covariate (log tokens): A and C. "
        "The Smell row comes from models with the binary Smell/Clean predictor; the other perceived-category "
        "rows come from models with the four (non-exclusive) perceived-type indicators.}",
        "\\label{tab:rq2_lmm}",
        "\\centering\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Term & A & B & C \\\\",
        "\\midrule",
    ]
    for term in order:
        lines.append(labels[term] + " & " + " & ".join(cell(re, size, term) for re, size, _ in specs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / "rq2_lmm_table.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
