import  numpy as np
import  scipy.stats as scs # beta, norm, lognorm, multivariate_normal
from    typing import List, Optional, Tuple, Union

calc_cosine_similarity = (
    lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if np.linalg.norm(a) != 0 and np.linalg.norm(b) != 0
    else 0.0
)
calc_cosine_similarity_matrix = lambda embeddings: embeddings @ embeddings.T

def calc_pct_change(base_value: float, new_value: float) -> float:
    """Compute the percent change from `base_value` to `new_value`.

    Args:
        base_value: The original/reference value.
        new_value: The new value being compared against `base_value`.

    Returns:
        The percent change from `base_value` to `new_value` (negative if it decreased).

    Raises:
        ValueError: If `base_value` is zero.

    Example:
        Compare latency between two agent runs (e.g. consecutive
        `ResponseStats.latency_sec` values from `Agent.run()`):

        >>> baseline_latency = 1.85
        >>> optimized_latency = 1.20
        >>> calc_pct_change(baseline_latency, optimized_latency)
        -35.135135135135144
    """
    try:
        return ((new_value - base_value) / base_value) * 100
    except ZeroDivisionError as e:
        raise ValueError("base_value must be non-zero to calculate percent change") from e


def calc_weighted_avg(values: List[float], weights: List[float]) -> float:
    """Compute the weighted average of `values`, weighted by `weights`.

    Args:
        values: The values to average.
        weights: The weight for each value, same length as `values`.

    Returns:
        The weighted average of `values`.

    Raises:
        ValueError: If `values` and `weights` have different lengths, or if `weights` sum to zero.

    Example:
        Weight by sample size — averaging a metric across groups of
        different sizes, where larger groups should count more. Weights
        can be retrieved directly from a groupby count:

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "batch":    ["a", "a", "b", "b", "b", "b"],
        ...     "accuracy": [0.90, 0.92, 0.75, 0.78, 0.74, 0.77],
        ... })
        >>> grouped = df.groupby("batch")["accuracy"].mean()
        >>> weights = df.groupby("batch").size()  # sample count per batch
        >>> calc_weighted_avg(grouped.tolist(), weights.tolist())
        0.81

        Weight by token count — averaging latency across agent runs by how
        much work each run actually did (e.g. `ResponseStats.usage_stats`
        input + output tokens from `mcp_client.py`):

        >>> latencies = [1.2, 2.5, 0.8]
        >>> tokens    = [500, 1500, 200]
        >>> calc_weighted_avg(latencies, tokens)
        2.05

        Weight by recency — more recent observations count more, via
        exponential decay:

        >>> scores = [0.70, 0.75, 0.90]  # oldest to newest
        >>> weights = [0.5 ** i for i in range(len(scores) - 1, -1, -1)]
        >>> weights
        [0.25, 0.5, 1.0]
        >>> calc_weighted_avg(scores, weights)
        0.8285714285714285
    """
    # e.g., equivalent to np.average(values,weights=weights)
    try:
        weighted_sum = sum(v * w for v, w in zip(values, weights, strict=True))
        return weighted_sum / sum(weights)
    except ValueError as e:
        raise ValueError(f"values and weights must be the same length (got {len(values)} and {len(weights)})") from e
    except ZeroDivisionError as e:
        raise ValueError("sum of weights must be non-zero to calculate weighted average") from e


def calc_z_score(
    values: Union[float, List[float], np.ndarray],
    mean:   Optional[float] = None,
    std:    Optional[float] = None,
    ddof:   int = 0,
) -> Union[float, np.ndarray]:
    """Compute z-scores: how many standard deviations `values` sit from the mean.

    Args:
        values (Union[float, List[float], np.ndarray]): Value(s) to standardize.
        mean (Optional[float]): Reference mean; defaults to the mean of `values`.
        std (Optional[float]): Reference standard deviation; defaults to the std of `values`.
        ddof (int): Delta degrees of freedom for the default std; 0 = population, 1 = sample.

    Returns:
        Union[float, np.ndarray]: A float for a scalar input; an array matching `values` otherwise.

    Raises:
        ValueError: If the standard deviation is zero.

    Examples:
        >>> latencies = [1.2, 1.4, 1.1, 1.3, 4.8]    # per-run latency_sec across agent runs
        >>> z_scores  = calc_z_score(latencies)      # standardize within the batch
        >>> outliers  = [x for x, z in zip(latencies, z_scores) if abs(z) > 1.5]
        >>> calc_z_score(3.0, mean=1.0, std=0.5)     # score a new run against a stored baseline
        4.0
    """
    arr    = np.asarray(values, dtype=np.float64)
    n, dim = arr.size, arr.ndim

    mean = arr.mean() if mean is None else mean
    std  = arr.std(ddof=ddof) if std is None else std
    if std == 0:
        raise ValueError("std must be non-zero to calculate z-score")
    result = (arr - mean) / std
    return float(result) if result.ndim == 0 else result


def calc_confidence_interval(values: Union[List[float], np.ndarray], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute a confidence interval for the mean of `values`: `mean ± z_crit * se`.

    Works on any values, e.g. the output of `calc_z_score()` (bounds are then on the z-scale; convert to
    raw units with `mean + bound * std`). `z_crit` is the z-score cutoff for the confidence level
    (1.645 at 90%, 1.96 at 95%, 2.576 at 99%).

    Args:
        values (Union[List[float], np.ndarray]): Observations to compute the interval for.
        confidence (float): Confidence level in (0, 1), e.g. 0.95 for a 95% interval.

    Returns:
        Tuple[float, float]: `(lower, upper)` bounds of the interval.

    Raises:
        ValueError: If `confidence` is not in (0, 1) or there are fewer than 2 values.

    Examples:
        >>> new_latencies = [1.5, 1.6, 1.4, 1.5, 1.7]   # latest batch of agent runs (latency_sec)
        >>> z_scores = calc_z_score(new_latencies, mean=1.30, std=0.13)   # against a stored baseline
        >>> lower, upper = calc_confidence_interval(z_scores)
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        raise ValueError(f"at least 2 values are required, got {arr.size}")
    z_crit   = scs.norm.ppf(1 - (1 - confidence) / 2)
    margin   = z_crit * scs.sem(arr)
    ci_lower = float(arr.mean() - margin)
    ci_upper = float(arr.mean() + margin)
    return ci_lower, ci_upper
