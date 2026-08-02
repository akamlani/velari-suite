import  numpy as np
from    typing import List, Literal, Union
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
