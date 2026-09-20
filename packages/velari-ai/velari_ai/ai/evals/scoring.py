import  numpy as np
import  editdistance as ed
from    typing import Any, List, Literal, Sequence, Union
from    scipy import spatial

# cdist: cross-distance (inter),    compute distance between each pair of the two collections of inputs
# pdist: pairwise-distance (intra), compute pairwise distance between every pair of rows within one collection


def _pairwise_distance(
    a: Union[List[float], np.ndarray],
    b: Union[List[float], np.ndarray],
    metric: Literal["euclidean", "cosine"],
) -> Union[float, np.ndarray]:
    """Compute a pairwise distance metric between vectors and/or matrices via scipy's cdist."""
    a_arr  = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b_arr  = np.atleast_2d(np.asarray(b, dtype=np.float64))
    result = spatial.distance.cdist(a_arr, b_arr, metric=metric).squeeze()
    return float(result) if result.ndim == 0 else result

def _condensed_distance(x: Union[List[List[float]], np.ndarray], metric: Literal["euclidean", "cosine"]) -> np.ndarray:
    """Compute condensed pairwise distances within one collection via scipy's pdist."""
    x_arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return spatial.distance.pdist(x_arr, metric=metric)


def euclidean_distance(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
    """Compute Euclidean (L2) distance between embedding vectors and/or matrices.

    Args:
        a (Union[List[float], np.ndarray]): Query embedding vector, or an (n, d) embedding matrix.
        b (Union[List[float], np.ndarray]): Document embedding vector, or an (n, d) embedding matrix.

    Returns:
        Union[float, np.ndarray]: A float when both inputs are single vectors; 0.0 when identical.
            An array of per-row distances when either input is a matrix.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]
        >>> query_embedding   = [0.1, 0.8, 0.3, 0.5]
        >>> scores = [euclidean_distance(query_embedding, doc) for doc in corpus_embeddings]
        >>> scores_batch = euclidean_distance(corpus_embeddings, query_embedding)  # vectorized, same values as `scores`
    """
    return _pairwise_distance(a, b, metric="euclidean")

def cosine_distance(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
    """Compute cosine distance between embedding vectors and/or matrices.

    Args:
        a (Union[List[float], np.ndarray]): Query embedding vector, or an (n, d) embedding matrix.
        b (Union[List[float], np.ndarray]): Document embedding vector, or an (n, d) embedding matrix.

    Returns:
        Union[float, np.ndarray]: A float in [0, 2] when both inputs are single vectors
            (0.0 = identical direction, 1.0 = orthogonal, 2.0 = opposite). An array of
            per-row distances when either input is a matrix.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]
        >>> query_embedding   = [0.1, 0.8, 0.3, 0.5]
        >>> scores = [cosine_distance(query_embedding, doc) for doc in corpus_embeddings]
        >>> scores_batch = cosine_distance(corpus_embeddings, query_embedding)  # vectorized, same values as `scores`
    """
    return _pairwise_distance(a, b, metric="cosine")

def cosine_similarity(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
    """Compute cosine similarity between embedding vectors and/or matrices.

    Args:
        a (Union[List[float], np.ndarray]): Query embedding vector, or an (n, d) embedding matrix.
        b (Union[List[float], np.ndarray]): Document embedding vector, or an (n, d) embedding matrix.

    Returns:
        Union[float, np.ndarray]: A float in [-1, 1] when both inputs are single vectors
            (1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite). An array of
            per-row similarities when either input is a matrix.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]
        >>> query_embedding   = [0.1, 0.8, 0.3, 0.5]
        >>> scores = [cosine_similarity(query_embedding, doc) for doc in corpus_embeddings]
        >>> scores_batch = cosine_similarity(corpus_embeddings, query_embedding)  # vectorized, same values as `scores`
    """
    return 1.0 - cosine_distance(a, b)

def intra_euclidean_distance(x: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Compute condensed pairwise Euclidean (L2) distances within one embedding matrix.

    Args:
        x (Union[List[List[float]], np.ndarray]): Embedding matrix, shape (n, d).

    Returns:
        np.ndarray: Condensed pairwise distances, length n*(n-1)/2 (upper triangle only,
            no diagonal/self-distances) — see scipy.spatial.distance.squareform() to expand.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1], [0.1, 0.9, 0.2, 0.4]]
        >>> distances = intra_euclidean_distance(corpus_embeddings)
    """
    return _condensed_distance(x, metric="euclidean")

def intra_cosine_distance(x: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Compute condensed pairwise cosine distances within one embedding matrix.

    Args:
        x (Union[List[List[float]], np.ndarray]): Embedding matrix, shape (n, d).

    Returns:
        np.ndarray: Condensed pairwise cosine distances in [0, 2], length n*(n-1)/2.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1], [0.1, 0.9, 0.2, 0.4]]
        >>> distances = intra_cosine_distance(corpus_embeddings)
    """
    return _condensed_distance(x, metric="cosine")

def intra_cosine_similarity(x: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Compute condensed pairwise cosine similarities within one embedding matrix.

    Args:
        x (Union[List[List[float]], np.ndarray]): Embedding matrix, shape (n, d).

    Returns:
        np.ndarray: Condensed pairwise cosine similarities in [-1, 1], length n*(n-1)/2.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1], [0.1, 0.9, 0.2, 0.4]]
        >>> similarities = intra_cosine_similarity(corpus_embeddings)
    """
    return 1.0 - intra_cosine_distance(x)

def edit_distance(output: Sequence[Any], expected: Sequence[Any]) -> int:
    """Compute the edit distance between the `output` and `expected`.

    Args:
        output (Sequence[Any]): The output to compare (a string, or any sequence of hashable items).
        expected (Sequence[Any]): The expected output to compare against.

    Returns:
        int: The edit distance between the `output` and `expected`.

    Examples:
        >>> output   = json.dumps({"answer": "42"}, sort_keys=True)
        >>> expected = json.dumps({"answer": "43"}, sort_keys=True)
        >>> distance = edit_distance(output, expected)
    """
    return ed.eval(output, expected)

def cross_entropy(values: Union[Sequence[float], np.ndarray], is_logprob: bool = False) -> float:
    """Compute cross-entropy (mean negative log-likelihood, natural log) of the tokens a model assigned to a text.

    Log-scale form of `perplexity()` (`perplexity = exp(cross_entropy)`). Prefer it for math on the
    number: averaging across documents, tracking eval loss, or comparing small model differences.
    Measures confidence, not correctness; only comparable across models sharing a tokenizer.

    Args:
        values (Union[Sequence[float], np.ndarray]): 1-D probabilities of the true tokens in (0, 1], or natural-log probabilities if `is_logprob`.
        is_logprob (bool): Set True for API-style `logprobs`; avoids a lossy exp/log round trip.

    Returns:
        float: Cross-entropy (natural log, unit "nats") >= 0.0; 0.0 = fully certain, larger = more uncertain.

    Raises:
        ValueError: If `values` is empty, or contains a non-positive probability.

    Examples:
        >>> client   = OpenAI()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Summarize the Q3 churn-analysis report."}],
        ...     logprobs=True,
        ... )
        >>> logprobs = [t.logprob for t in response.choices[0].logprobs.content]   # natural-log, generated tokens only
        >>> loss = cross_entropy(logprobs, is_logprob=True)
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if not is_logprob and np.any(arr <= 0):
        raise ValueError("probabilities must all be > 0")
    log_probs = arr if is_logprob else np.log(arr)
    return float(-np.mean(log_probs))

def perplexity(values: Union[Sequence[float], np.ndarray], is_logprob: bool = False) -> float:
    """Compute perplexity, the exponentiated average negative log-likelihood a model assigns to a text.

    Human-readable form of `cross_entropy()`: roughly "as unsure as choosing among PPL equally likely
    tokens per step". Prefer it for reporting: comparing models on a held-out corpus, or flagging
    low-confidence generations and drift. Use `cross_entropy()` to average or track loss.
    Measures confidence, not correctness; only comparable across models sharing a tokenizer.

    Computation, given per-token probabilities p_1..p_N:
        1. log_probs    = log(p)                 # skipped if `is_logprob`; values are already log(p)
        2. avg_log_prob = mean(log_probs)        # average over the N tokens
        3. perplexity   = exp(-avg_log_prob)     # step 2 negated is `cross_entropy()`
    The log base only has to match its inverse: `2 ** (-mean(log2(p)))` gives the same result.

    Args:
        values (Union[Sequence[float], np.ndarray]): 1-D per-token probabilities in (0, 1], or natural-log probabilities if `is_logprob`.
        is_logprob (bool): Set True for API-style `logprobs`; avoids a lossy exp/log round trip.

    Returns:
        float: Perplexity >= 1.0; 1.0 = fully certain, larger = more uncertain.

    Raises:
        ValueError: If `values` is empty, or contains a non-positive probability.

    Examples:
        >>> client   = OpenAI()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Draft a reply to support ticket TCK-2291."}],
        ...     logprobs=True,
        ... )
        >>> logprobs = [t.logprob for t in response.choices[0].logprobs.content]   # natural-log, generated tokens only
        >>> ppl = perplexity(logprobs, is_logprob=True)
        >>> ppl = perplexity([0.95, 0.73, 0.30, 0.98])   # same idea from raw token probabilities
    """
    return float(np.exp(cross_entropy(values, is_logprob=is_logprob)))
