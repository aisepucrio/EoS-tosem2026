"""Exclusion of implausibly long fixations.

The iTrace-Toolkit I-VT output contains a small number of extremely long
"fixations" (up to ~24 min) that cannot correspond to single physiological
fixations and most likely reflect periods in which gaze was not resolved
(e.g., the participant looked away or the tracker lost the eyes while the
I-VT filter kept merging samples). They are few (<1% of fixations) but carry
a large share of the total fixation time, so they are excluded before any
fixation-based metric is computed.

Because RQ3 classifies each fixation by comparing it with the *preceding*
fixation (Bottom-Up vs. Top-Down), the fixation that followed an excluded one
is re-classified against the previous retained fixation. All other original
labels are kept unchanged.
"""

from __future__ import annotations

import pandas as pd

MAX_FIXATION_MS = 5000

KEY = ["participant_id", "snippet_id"]
ORDER = KEY + ["fixation_run_id", "fixation_start_event_time"]


def exclude_long_fixations(
    fixations: pd.DataFrame,
    max_ms: float | None = MAX_FIXATION_MS,
    relabel_events: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return fixations with duration <= ``max_ms`` (sorted in temporal order).

    If ``relabel_events`` is True and an ``event``/``chunk`` column is present,
    the first retained fixation after an excluded one is re-classified as
    ``top-down`` when its chunk differs from that of the previous retained
    fixation in the same participant x snippet cell (``bottom-up`` otherwise;
    ``top-down`` if it becomes the first fixation of the cell).
    """

    data = fixations.sort_values(ORDER).reset_index(drop=True)

    if max_ms is None:
        return data

    duration = pd.to_numeric(data["duration"], errors="coerce")
    removed = duration.gt(max_ms)

    # Was the immediately preceding fixation (same cell) removed?
    prev_removed = removed.groupby([data[k] for k in KEY]).shift(fill_value=False)

    kept = data.loc[~removed].copy()

    if relabel_events and {"event", "chunk"}.issubset(kept.columns):
        needs_relabel = prev_removed.loc[~removed].astype(bool)
        prev_chunk = kept.groupby(KEY)["chunk"].shift()
        new_event = pd.Series("bottom-up", index=kept.index).where(
            kept["chunk"].eq(prev_chunk), "top-down"
        )
        kept.loc[needs_relabel, "event"] = new_event.loc[needs_relabel]
        n_relabel = int(needs_relabel.sum())
    else:
        n_relabel = 0

    if verbose:
        share_time = duration[removed].sum() / duration.sum()
        print(
            f"Excluded fixations > {max_ms:,} ms: {int(removed.sum()):,} of {len(data):,} "
            f"({removed.mean():.2%} of fixations, {share_time:.2%} of fixation time); "
            f"re-classified successors: {n_relabel}"
        )

    return kept.reset_index(drop=True)
