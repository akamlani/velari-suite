import  numpy as np
from    typing import Sequence, Union

def cross_entropy_dist(
    p: Union[Sequence[float], np.ndarray],
    q: Union[Sequence[float], np.ndarray],
    base: float = 2.0,
    eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """Compute cross-entropy H(p, q) between a target distribution `p` and a predicted distribution `q`.

    Use it when the target is a full distribution rather than one observed token: soft labels,
    label smoothing, distilling a teacher model into a student, or scoring a model's next-token
    distribution against a reference. For the one-hot case (scoring observed tokens from API
    `logprobs`), use `cross_entropy()` instead. Lower is better; it is minimized when `q == p`.

    Args:
        p (Union[Sequence[float], np.ndarray]): Target distribution(s); last axis sums to 1.
        q (Union[Sequence[float], np.ndarray]): Predicted distribution(s), same shape as `p`.
        base (float): Log base: 2.0 = log2 (unit "bits"); `np.e` = natural log (unit "nats", as in `cross_entropy()`).
        eps (float): Floor on `q` so a zero where `p > 0` yields a large finite value, not `inf`.

    Returns:
        Union[float, np.ndarray]: A float for 1-D inputs; one value per row for 2-D inputs.

    Raises:
        ValueError: If shapes differ, or a row of `p` or `q` doesn't sum to 1 (empty input included).

    Examples:
        >>> teacher = [0.70, 0.20, 0.05, 0.05]   # reference next-token distribution
        >>> student = [0.55, 0.30, 0.10, 0.05]   # fine-tuned model's distribution over the same 4 tokens
        >>> bits = cross_entropy_dist(teacher, student)                # log2
        >>> nats = cross_entropy_dist(teacher, student, base=np.e)     # natural log; bits = nats / ln(2)
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"p and q must have the same shape, got {p_arr.shape} and {q_arr.shape}")
    if not (np.allclose(p_arr.sum(axis=-1), 1.0) and np.allclose(q_arr.sum(axis=-1), 1.0)):
        raise ValueError("p and q must each sum to 1 along the last axis")
    result = -(p_arr * np.log(np.clip(q_arr, eps, 1.0))).sum(axis=-1) / np.log(base)
    return float(result) if result.ndim == 0 else result

def perplexity_dist(
    p: Union[Sequence[float], np.ndarray],
    q: Union[Sequence[float], np.ndarray],
    eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """Compute perplexity of predicted distribution `q` against target distribution `p`: exp(H(p, q)).

    Human-readable form of `cross_entropy_dist()`: the effective number of equally likely
    outcomes `q` is choosing among, judged against `p`. Use it to report how well a model's
    distribution matches a reference (e.g. student vs. teacher); it equals `exp(H(p))` when
    `q == p` and grows as `q` diverges. Base-independent, so no `base` argument.

    Args:
        p (Union[Sequence[float], np.ndarray]): Target distribution(s); last axis sums to 1.
        q (Union[Sequence[float], np.ndarray]): Predicted distribution(s), same shape as `p`.
        eps (float): Floor on `q` so a zero where `p > 0` yields a large finite value, not `inf`.

    Returns:
        Union[float, np.ndarray]: A float for 1-D inputs; one value per row for 2-D inputs.

    Raises:
        ValueError: If shapes differ, or a row of `p` or `q` doesn't sum to 1 (empty input included).

    Examples:
        >>> reference = [0.60, 0.25, 0.10, 0.05]   # annotator label distribution for a support-ticket category
        >>> predicted = [0.50, 0.30, 0.15, 0.05]   # classifier's predicted distribution
        >>> ppl = perplexity_dist(reference, predicted)
    """
    return np.exp(cross_entropy_dist(p, q, base=np.e, eps=eps))

def kl_divergence(
    p: Union[Sequence[float], np.ndarray],
    q: Union[Sequence[float], np.ndarray],
    base: float = 2.0,
    eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """Compute KL divergence D(p || q) = H(p, q) - H(p): the extra cost of using `q` in place of `p`.

    Use it instead of `cross_entropy_dist()` when you want a score that is 0 for a perfect
    match rather than floored at the target's own entropy H(p), e.g. comparing model outputs
    across prompts whose reference distributions differ in spread. Asymmetric: D(p || q) != D(q || p).

    Args:
        p (Union[Sequence[float], np.ndarray]): Target distribution(s); last axis sums to 1.
        q (Union[Sequence[float], np.ndarray]): Predicted distribution(s), same shape as `p`.
        base (float): Log base: 2.0 = log2 (unit "bits"); `np.e` = natural log (unit "nats").
        eps (float): Floor on `q` so a zero where `p > 0` yields a large finite value, not `inf`.

    Returns:
        Union[float, np.ndarray]: Divergence >= 0.0 (0.0 = identical); a float for 1-D inputs.

    Raises:
        ValueError: If shapes differ, or a row of `p` or `q` doesn't sum to 1 (empty input included).

    Examples:
        >>> baseline  = [0.40, 0.30, 0.20, 0.10]   # base model's token distribution
        >>> finetuned = [0.55, 0.25, 0.15, 0.05]   # same prompt after fine-tuning
        >>> drift = kl_divergence(finetuned, baseline)
    """
    return cross_entropy_dist(p, q, base=base, eps=eps) - cross_entropy_dist(p, p, base=base, eps=eps)
