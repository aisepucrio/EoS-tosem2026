"""
Re-derive the area-of-interest labels (chunk, inside/outside, bottom-up/top-down) of the fixations.

Usage (from the repository root):

    python metrics_update/analysis/relabel_aoi.py

Why
---
Two labeling problems were found in ``data/all_exp_fixations_original2_cleaned.csv``:

1. Participants 10 and 27: every snippet was labeled with the AOI definition of the *previous*
   snippet (e.g., snippet 4 fixations carry snippet 3's chunks, snippet 3 fixations are all "before"
   because snippet 2's code starts at line 74). Their line numbers are correct; only the labels are wrong.
2. Snippet 13 (all participants): every fixation is labeled "outside", and the label ``chunk3``
   (lines 957-958) has no ``chunk_num`` because ``after`` was numbered 3 as well.

How
---
The labeling rules are recovered from the 23 participants without problem 1 and checked against their
original labels (the script asserts full agreement outside snippet 13):

- chunks are fixed line ranges per snippet (``before``, ``chunk1..k``, ``after``);
- a fixation is *inside* if its chunk is ``chunk1..k``, *outside* if ``before``/``after``;
- a fixation is *top-down* if its chunk differs from the chunk of the previous fixation on the same
  snippet (the first fixation is top-down), *bottom-up* otherwise.

Lines not covered by any observed range take the chunk of the closest range above them.

Outputs
-------
- ``metrics_update/data/fixations_aoi_relabeled.csv``: the original file with ``chunk``,
  ``chunk_num`` and ``event`` re-derived, plus ``*_original`` copies and ``aoi_relabeled``.
- ``metrics_update/outputs/snippets/aoi_chunk_ranges.csv``: the AOI definition per snippet.
- ``metrics_update/outputs/snippets/aoi_relabel_changes.csv``: changes by participant and snippet.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
FIXATIONS_PATH = ROOT_DIR / "data" / "all_exp_fixations_original2_cleaned.csv"
OUTPUT_PATH = ROOT_DIR / "metrics_update" / "data" / "fixations_aoi_relabeled.csv"
REPORT_DIR = ROOT_DIR / "metrics_update" / "outputs" / "snippets"

MISLABELED_PARTICIPANTS = [10, 27]
MISLABELED_SNIPPETS = [13]

ORDER = ["experiment", "fixation_target", "fixation_start_event_time"]


def chunk_rank(label: str) -> int:
    """before -> 0, chunkN -> N, after -> large (sorted last)."""
    if label == "before":
        return 0
    if label == "after":
        return 10_000

    return int(re.fullmatch(r"chunk(\d+)", label).group(1))


def chunk_ranges(reference: pd.DataFrame) -> pd.DataFrame:
    """Observed line range of every chunk label, per snippet."""
    ranges = (
        reference.groupby(["fixation_target", "chunk"])["source_file_line"]
        .agg(first_line="min", last_line="max", n_fixations="size")
        .reset_index()
    )
    ranges["rank"] = ranges["chunk"].map(chunk_rank)
    ranges = ranges.sort_values(["fixation_target", "rank"]).reset_index(drop=True)

    # Observed ranges of consecutive chunks must not overlap.
    for snippet, group in ranges.groupby("fixation_target"):
        assert (group["first_line"].iloc[1:].to_numpy() > group["last_line"].iloc[:-1].to_numpy()).all(), snippet

    k = ranges.loc[ranges["chunk"].str.startswith("chunk")].groupby("fixation_target")["rank"].max()
    ranges["chunk_num"] = np.where(
        ranges["chunk"].eq("after"), ranges["fixation_target"].map(k) + 1, ranges["rank"]
    ).astype(int)
    ranges["inside"] = ranges["chunk"].str.startswith("chunk")

    return ranges.drop(columns="rank")


def assign_chunks(fixations: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Chunk of each fixation: the last range whose first line is <= the fixation line."""
    parts = []

    for snippet, group in fixations.groupby("fixation_target", sort=False):
        table = ranges.loc[ranges["fixation_target"].eq(snippet)].reset_index(drop=True)
        position = np.searchsorted(table["first_line"].to_numpy(), group["source_file_line"].to_numpy(), side="right") - 1
        position = position.clip(0)  # lines above the first observed range -> "before"
        parts.append(
            pd.DataFrame(
                {
                    "chunk": table["chunk"].to_numpy()[position],
                    "chunk_num": table["chunk_num"].to_numpy()[position],
                    "inside": table["inside"].to_numpy()[position],
                },
                index=group.index,
            )
        )

    return pd.concat(parts).loc[fixations.index]


def derive_event(fixations: pd.DataFrame) -> pd.Series:
    ordered = fixations.sort_values(ORDER)
    previous = ordered.groupby(["experiment", "fixation_target"])["chunk"].shift()
    direction = np.where(ordered["chunk"].ne(previous), "top-down", "bottom-up")
    location = np.where(ordered["inside"], "inside", "outside")

    return pd.Series(np.char.add(np.char.add(direction, "-"), location), index=ordered.index).loc[fixations.index]


def main() -> None:
    fixations = pd.read_csv(FIXATIONS_PATH, sep="\t")
    fixations = fixations.sort_values(ORDER).reset_index(drop=True)

    reference = fixations.loc[~fixations["experiment"].isin(MISLABELED_PARTICIPANTS)]
    ranges = chunk_ranges(reference)

    relabeled = assign_chunks(fixations, ranges)
    relabeled["event"] = derive_event(fixations.assign(chunk=relabeled["chunk"], inside=relabeled["inside"]))

    # Validation: on the reference participants, the re-derived labels reproduce the original ones.
    check = ~fixations["experiment"].isin(MISLABELED_PARTICIPANTS)
    assert relabeled.loc[check, "chunk"].eq(fixations.loc[check, "chunk"]).all()
    check_event = check & ~fixations["fixation_target"].isin(MISLABELED_SNIPPETS)
    assert relabeled.loc[check_event, "event"].eq(fixations.loc[check_event, "event"]).all()
    print(f"Validation: chunk labels reproduced for {check.sum():,} fixations; "
          f"event labels reproduced for {check_event.sum():,} fixations (snippet 13 excluded).")

    output = fixations.copy()
    for column in ("chunk", "chunk_num", "event"):
        output[f"{column}_original"] = fixations[column]
        output[column] = relabeled[column]
    output["aoi_relabeled"] = (
        output["chunk"].ne(output["chunk_original"])
        | output["chunk_num"].ne(output["chunk_num_original"])
        | output["event"].ne(output["event_original"])
    )

    changes = (
        output.assign(
            chunk_changed=output["chunk"].ne(output["chunk_original"]),
            location_changed=output["event"].str.split("-").str[-1].ne(output["event_original"].str.split("-").str[-1]),
            direction_changed=output["event"].str.rsplit("-", n=1).str[0].ne(output["event_original"].str.rsplit("-", n=1).str[0]),
        )
        .groupby(["experiment", "fixation_target"])[["chunk_changed", "location_changed", "direction_changed"]]
        .agg(["sum"])
        .droplevel(1, axis=1)
        .assign(n_fixations=output.groupby(["experiment", "fixation_target"]).size())
        .reset_index()
    )
    changes = changes.loc[changes[["chunk_changed", "location_changed", "direction_changed"]].sum(axis=1).gt(0)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, sep="\t", index=False)
    ranges.to_csv(REPORT_DIR / "aoi_chunk_ranges.csv", index=False)
    changes.to_csv(REPORT_DIR / "aoi_relabel_changes.csv", index=False)

    print()
    print(changes.to_string(index=False))
    print()
    print("Fixations relabeled:", int(output["aoi_relabeled"].sum()), "of", len(output))
    for path in (OUTPUT_PATH, REPORT_DIR / "aoi_chunk_ranges.csv", REPORT_DIR / "aoi_relabel_changes.csv"):
        print("Saved:", path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
