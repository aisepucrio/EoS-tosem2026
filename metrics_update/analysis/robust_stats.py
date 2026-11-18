from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

PairLabels = tuple[str, str]
StatisticalResult = dict[str, int | float]


def _resolve_pair(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    smell: ArrayLike | None = None,
    clean: ArrayLike | None = None,
    group_a: ArrayLike | None = None,
    group_b: ArrayLike | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    PairLabels | None,
]:
    """
    Normalize the supported input names into a single pair of numeric arrays.

    Exactly one of the following pairs must be provided:
    (x, y), (smell, clean), or (group_a, group_b).

    Parameters
    ----------
    x, y
        Generic names for the two paired conditions.
    smell, clean
        Alternative names used when comparing smell and clean conditions.
    group_a, group_b
        Alternative generic names for two paired groups.

    Returns
    -------
    tuple
        A tuple containing:
        - the first condition as a NumPy float array;
        - the second condition as a NumPy float array;
        - inferred labels for the conditions, or None when x/y are used.

    Raises
    ------
    ValueError
        If only one element of a pair is provided, more than one pair is
        provided, no pair is provided, or the two arrays have different shapes.
    """

    sources: list[
        tuple[
            ArrayLike,
            ArrayLike,
            PairLabels | None,
        ]
    ] = []

    if x is not None or y is not None:
        if x is None or y is None:
            raise ValueError("Both x and y must be provided if either is provided.")

        sources.append((x, y, None))

    if smell is not None or clean is not None:
        if smell is None or clean is None:
            raise ValueError(
                "Both smell and clean must be provided if either is provided."
            )

        sources.append((smell, clean, ("smell", "clean")))

    if group_a is not None or group_b is not None:
        if group_a is None or group_b is None:
            raise ValueError(
                "Both group_a and group_b must be provided if either is provided."
            )

        sources.append((group_a, group_b, ("group_a", "group_b")))

    if len(sources) != 1:
        raise ValueError(
            "Exactly one pair must be provided: "
            "(x, y), (smell, clean), or (group_a, group_b)."
        )

    a, b, inferred_labels = sources[0]

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            "The two paired conditions must have the same shape. "
            f"Got {a.shape} and {b.shape}."
        )

    return a, b, inferred_labels


def yuen_paired(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    trim: float = 0.20,
    alpha: float = 0.05,
    *,
    smell: ArrayLike | None = None,
    clean: ArrayLike | None = None,
    group_a: ArrayLike | None = None,
    group_b: ArrayLike | None = None,
    percent_points: bool = False,
) -> StatisticalResult:
    """
    Perform Yuen's robust test for two paired/dependent conditions.

    The implementation follows the formulation used by WRS2::yuend():
    trimmed means are computed separately for the two conditions, while
    dependence between paired observations is incorporated through the
    Winsorized covariance.

    This is not equivalent to applying a one-sample trimmed-mean test to
    the vector of pairwise differences.

    Parameters
    ----------
    x, y
        Generic names for the two paired conditions.
    trim
        Proportion trimmed from each tail. Must be in [0, 0.5).
        The default is 0.20.
    alpha
        Significance level used to compute the confidence interval.
        The default is 0.05.
    smell, clean
        Alternative names for x and y when comparing smell and clean
        conditions.
    group_a, group_b
        Alternative names for x and y when comparing generic paired groups.
    percent_points
        If True, also return the trimmed mean difference and confidence
        interval multiplied by 100.

    Returns
    -------
    dict
        Dictionary containing the sample size, trimmed mean difference,
        standard error, t statistic, degrees of freedom, p-value, and
        confidence interval.

    Raises
    ------
    ValueError
        If trim or alpha are invalid, too few observations remain after
        trimming, or the estimated squared standard error is negative.
    """

    x, y, _ = _resolve_pair(
        x,
        y,
        smell=smell,
        clean=clean,
        group_a=group_a,
        group_b=group_b,
    )

    if not (0 <= trim < 0.5):
        raise ValueError("trim must be in the interval [0, 0.5).")

    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")

    # Keep only pairs for which both observations are finite.
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    n = len(x)

    g = int(np.floor(trim * n))
    h = n - 2 * g

    if h < 2:
        raise ValueError("Too few observations remain after trimming.")

    def _trimmed_mean_wrs2(
        values: NDArray[np.float64],
    ) -> float:
        """
        Compute the trimmed mean following the WRS2 formulation.
        """

        ordered = np.sort(values)

        if g == 0:
            return float(ordered.mean())

        return float(ordered[g : n - g].mean())

    def _winsorize_wrs2(
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Winsorize values using the trimming limits from WRS2.

        Extreme observations are replaced by the nearest observations
        retained after trimming. The original observation order is
        preserved so that paired covariance can be computed correctly.
        """

        ordered = np.sort(values)

        lower = ordered[g]
        upper = ordered[n - g - 1]

        return np.clip(
            values,
            lower,
            upper,
        )

    trimmed_mean_x = _trimmed_mean_wrs2(x)
    trimmed_mean_y = _trimmed_mean_wrs2(y)

    trimmed_mean_diff = trimmed_mean_x - trimmed_mean_y

    winsorized_x = _winsorize_wrs2(x)
    winsorized_y = _winsorize_wrs2(y)

    # ddof=1 reproduces the n - 1 denominator used by
    # R::var() and R::cov().
    winsorized_variance_x = float(
        np.var(
            winsorized_x,
            ddof=1,
        )
    )

    winsorized_variance_y = float(
        np.var(
            winsorized_y,
            ddof=1,
        )
    )

    winsorized_covariance = float(
        np.cov(
            winsorized_x,
            winsorized_y,
            ddof=1,
        )[0, 1]
    )

    # Parameterization equivalent to WRS2::yuend().
    q1 = (n - 1) * winsorized_variance_x
    q2 = (n - 1) * winsorized_variance_y
    q3 = (n - 1) * winsorized_covariance

    se_squared = (q1 + q2 - 2 * q3) / (h * (h - 1))

    # Protect against very small negative values caused only by
    # floating-point rounding.
    if se_squared < 0 and np.isclose(se_squared, 0.0):
        se_squared = 0.0

    if se_squared < 0:
        raise ValueError(
            "The estimated squared standard error is negative. " "Check the input data."
        )

    se = float(np.sqrt(se_squared))

    df = h - 1

    if np.isclose(se, 0.0):
        if np.isclose(trimmed_mean_diff, 0.0):
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = float(np.sign(trimmed_mean_diff) * np.inf)
            p_value = 0.0

    else:
        t_stat = float(trimmed_mean_diff / se)

        p_value = float(
            2
            * stats.t.sf(
                np.abs(t_stat),
                df=df,
            )
        )

    critical = float(
        stats.t.ppf(
            1 - alpha / 2,
            df=df,
        )
    )

    ci_low = float(trimmed_mean_diff - critical * se)

    ci_high = float(trimmed_mean_diff + critical * se)

    result: StatisticalResult = {
        "n": n,
        "trimmed_mean_diff": trimmed_mean_diff,
        "se": se,
        "t": t_stat,
        "df": df,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

    if percent_points:
        result.update(
            {
                "trimmed_mean_diff_pp": (100 * trimmed_mean_diff),
                "ci_low_pp": (100 * ci_low),
                "ci_high_pp": (100 * ci_high),
            }
        )

    return result


def paired_phat(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    smell: ArrayLike | None = None,
    clean: ArrayLike | None = None,
    group_a: ArrayLike | None = None,
    group_b: ArrayLike | None = None,
    label_x: str | None = None,
    label_y: str | None = None,
) -> StatisticalResult:
    """
    Compute the paired probability of superiority (p-hat).

    For each valid pair:

    - x > y contributes 1.0;
    - x = y contributes 0.5;
    - x < y contributes 0.0.

    The paired p-hat is the arithmetic mean of these pairwise scores.

    Parameters
    ----------
    x, y
        Generic names for the two paired conditions.
    smell, clean
        Alternative names for x and y when comparing smell and clean
        conditions.
    group_a, group_b
        Alternative names for x and y when comparing generic paired groups.
    label_x, label_y
        Optional labels used in the names of the returned count fields.
        Both labels must be provided together.

    Returns
    -------
    dict
        Dictionary containing the number of valid pairs, p-hat, and counts
        of pairs in which x is greater than, equal to, or smaller than y.

    Raises
    ------
    ValueError
        If no valid pairs are available or only one custom label is provided.
    """

    x, y, inferred_labels = _resolve_pair(
        x,
        y,
        smell=smell,
        clean=clean,
        group_a=group_a,
        group_b=group_b,
    )

    valid = np.isfinite(x) & np.isfinite(y)

    diff = x[valid] - y[valid]

    if len(diff) == 0:
        raise ValueError("No valid pairs are available to compute p-hat.")

    scores = np.where(
        diff > 0,
        1.0,
        np.where(
            diff < 0,
            0.0,
            0.5,
        ),
    )

    if label_x is None and label_y is None:
        if inferred_labels is None:
            label_x, label_y = "x", "y"
        else:
            label_x, label_y = inferred_labels

    elif label_x is None or label_y is None:
        raise ValueError("label_x and label_y must be provided together.")

    return {
        "n": len(diff),
        "phat": float(scores.mean()),
        f"n_{label_x}_gt_{label_y}": int((diff > 0).sum()),
        "n_equal": int((diff == 0).sum()),
        f"n_{label_x}_lt_{label_y}": int((diff < 0).sum()),
    }


def paired_phat_bootstrap_ci(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260101,
    *,
    smell: ArrayLike | None = None,
    clean: ArrayLike | None = None,
    group_a: ArrayLike | None = None,
    group_b: ArrayLike | None = None,
) -> StatisticalResult:
    """
    Compute a percentile bootstrap confidence interval for paired p-hat.

    The resampling unit is the complete participant/pair, preserving the
    dependence between the two conditions.

    Parameters
    ----------
    x, y
        Generic names for the two paired conditions.
    n_boot
        Number of bootstrap samples.
    alpha
        Significance level. For example, alpha=0.05 produces a 95%
        confidence interval.
    seed
        Seed used by NumPy's random number generator.
    smell, clean
        Alternative names for x and y when comparing smell and clean
        conditions.
    group_a, group_b
        Alternative names for x and y when comparing generic paired groups.

    Returns
    -------
    dict
        Dictionary containing the lower and upper percentile bootstrap
        confidence limits and the number of bootstrap samples.

    Raises
    ------
    ValueError
        If n_boot is smaller than 1, alpha is invalid, or fewer than two
        valid pairs are available.
    """

    x, y, _ = _resolve_pair(
        x,
        y,
        smell=smell,
        clean=clean,
        group_a=group_a,
        group_b=group_b,
    )

    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")

    valid = np.isfinite(x) & np.isfinite(y)

    diff = x[valid] - y[valid]

    n = len(diff)

    if n < 2:
        raise ValueError("At least two valid pairs are required for the bootstrap.")

    scores = np.where(
        diff > 0,
        1.0,
        np.where(
            diff < 0,
            0.0,
            0.5,
        ),
    )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        n,
        size=(n_boot, n),
    )

    bootstrap_phat = scores[indices].mean(axis=1)

    ci_low, ci_high = np.quantile(
        bootstrap_phat,
        [
            alpha / 2,
            1 - alpha / 2,
        ],
    )

    return {
        "phat_ci_low": float(ci_low),
        "phat_ci_high": float(ci_high),
        "n_boot": n_boot,
    }


def benjamini_hochberg(
    p_values: Iterable[float],
) -> NDArray[np.float64]:
    """
    Apply the Benjamini-Hochberg false discovery rate correction.

    The correction is computed only for finite p-values. Non-finite
    positions are preserved as NaN in the returned array.

    Parameters
    ----------
    p_values
        Iterable containing raw p-values.

    Returns
    -------
    numpy.ndarray
        Array containing Benjamini-Hochberg adjusted p-values.
        Positions corresponding to non-finite input values remain NaN.

    Raises
    ------
    ValueError
        If any finite p-value lies outside the interval [0, 1].
    """

    p = np.asarray(
        list(p_values),
        dtype=float,
    )

    result = np.full(
        p.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p)

    pv = p[valid]

    if len(pv) == 0:
        return result

    if np.any((pv < 0) | (pv > 1)):
        raise ValueError("p-values must be in the interval [0, 1].")

    order = np.argsort(pv)

    ranked = pv[order]

    m = len(ranked)

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    # Enforce the monotonicity required by the
    # Benjamini-Hochberg procedure.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    reordered = np.empty(
        m,
        dtype=float,
    )

    reordered[order] = adjusted

    result[valid] = reordered

    return result
