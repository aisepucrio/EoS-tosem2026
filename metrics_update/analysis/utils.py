from __future__ import annotations

from typing import Hashable

import pandas as pd


def participant_group_summary(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    *,
    participant_col: str = "participant_id",
    snippet_col: str = "snippet_id",
    mean_col: str = "mean_value",
    median_col: str = "median_value",
) -> pd.DataFrame:
    """
    Aggregate a metric at the participant × group level.

    This function should be applied before inferential analyses so that
    each participant contributes a single summary value to each group.

    Parameters
    ----------
    data
        Input DataFrame containing participant-, group-, snippet-, and
        metric-level observations.
    group_col
        Column defining the groups being compared.
    metric_col
        Column containing the metric to summarize.
    participant_col
        Column containing participant identifiers.
    snippet_col
        Column containing snippet identifiers.
    mean_col
        Name of the output column containing the mean metric value.
    median_col
        Name of the output column containing the median metric value.

    Returns
    -------
    pd.DataFrame
        One row per participant × group, containing the mean and median
        metric values and the number of unique snippets contributing to
        the summary.
    """

    return (
        data.dropna(subset=[group_col, metric_col])
        .groupby(
            [participant_col, group_col],
            as_index=False,
        )
        .agg(
            **{
                mean_col: (metric_col, "mean"),
                median_col: (metric_col, "median"),
                "n_snippets": (snippet_col, "nunique"),
            }
        )
    )


def descriptive_by_group(
    participant_table: pd.DataFrame,
    group_col: str,
    *,
    value_col: str = "mean_value",
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Compute descriptive statistics across participants for each group.

    The input table is expected to contain one observation per
    participant × group, typically produced by
    ``participant_group_summary``.

    Parameters
    ----------
    participant_table
        DataFrame aggregated at the participant × group level.
    group_col
        Column defining the groups.
    value_col
        Column containing the participant-level values to summarize.
    participant_col
        Column containing participant identifiers.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics for each group, including the number of
        participants, mean, median, standard deviation, first quartile,
        and third quartile.
    """

    return participant_table.groupby(
        group_col,
        as_index=False,
    ).agg(
        n_participants=(
            participant_col,
            "nunique",
        ),
        mean=(
            value_col,
            "mean",
        ),
        median=(
            value_col,
            "median",
        ),
        sd=(
            value_col,
            "std",
        ),
        q1=(
            value_col,
            lambda x: x.quantile(0.25),
        ),
        q3=(
            value_col,
            lambda x: x.quantile(0.75),
        ),
    )


def build_group_pairs(
    participant_table: pd.DataFrame,
    group_col: str,
    group_a: Hashable,
    group_b: Hashable,
    *,
    value_col: str = "mean_value",
    participant_col: str = "participant_id",
    count_col: str = "n_snippets",
) -> pd.DataFrame:
    """
    Build complete participant-level pairs for two groups.

    The input table must already be aggregated at the
    participant × group level. Only participants with observations
    in both groups are retained.

    Parameters
    ----------
    participant_table
        DataFrame containing one row per participant × group.
    group_col
        Column defining the groups being compared.
    group_a
        First group in the comparison.
    group_b
        Second group in the comparison.
    value_col
        Participant-level metric used in the comparison.
    participant_col
        Column containing participant identifiers.
    count_col
        Column containing the number of unique snippets contributing
        to each participant × group summary.

    Returns
    -------
    pd.DataFrame
        One row per participant with values for groups A and B,
        their difference, the number of snippets contributing to each
        group, and a comparison label.
    """

    selected = participant_table.loc[
        participant_table[group_col].isin([group_a, group_b])
    ].copy()

    values = selected.pivot(
        index=participant_col,
        columns=group_col,
        values=value_col,
    )

    if group_a not in values.columns or group_b not in values.columns:
        return pd.DataFrame(
            columns=[
                participant_col,
                "value_a",
                "value_b",
                "diff",
                "n_snippets_a",
                "n_snippets_b",
                "comparison",
            ]
        )

    pairs = (
        values[[group_a, group_b]]
        .dropna()
        .rename(
            columns={
                group_a: "value_a",
                group_b: "value_b",
            }
        )
    )

    if count_col in selected.columns:
        counts = selected.pivot(
            index=participant_col,
            columns=group_col,
            values=count_col,
        )

        pairs["n_snippets_a"] = counts.loc[
            pairs.index,
            group_a,
        ]

        pairs["n_snippets_b"] = counts.loc[
            pairs.index,
            group_b,
        ]

    else:
        pairs["n_snippets_a"] = pd.NA
        pairs["n_snippets_b"] = pd.NA

    pairs["diff"] = pairs["value_a"] - pairs["value_b"]

    pairs["comparison"] = f"{group_a} vs {group_b}"

    return pairs.reset_index()
