import  numpy as np
import  scipy.stats as scs # beta, norm, lognorm, multivariate_normal
from    typing import List

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
        ValueError: If `values` and `weights` have different lengths, or if
            `weights` sum to zero.

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
