from __future__ import annotations
import  numpy as np
from    typing               import Any, Optional, Union
from    sklearn.metrics.pairwise import cosine_similarity


def _to_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(1, -1) if arr.ndim < 2 else arr.reshape(arr.shape[0], -1)


def _from_matrix(scores: np.ndarray, a: Any, b: Optional[Any] = None) -> Union[float, np.ndarray]:
    """Reshape a `cosine_scores` matrix back to the sample axes of its inputs; 1D inputs drop their axis.

    Args:
        scores (np.ndarray): Similarity matrix of shape (n_a, n_b) from `cosine_scores(a, b)`.
        a (Any): The original first input passed to `cosine_scores`.
        b (Optional[Any]): The original second input; defaults to `a`.

    Returns:
        Union[float, np.ndarray]: Shape `(*a_samples, *b_samples)` — a plain `float` for 1D/1D, (n_b,) for 1D/2D, (n_a, n_b) for 2D/2D.

    Examples:
        >>> query  = [0.1, 0.8, 0.3, 0.5]
        >>> corpus = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]
        >>> _from_matrix(cosine_scores(query, corpus), query, corpus)
        array([0.981, 0.586])
        >>> _from_matrix(cosine_scores(query, query), query, query)
        1.0
    """
    shapes = (np.shape(a), np.shape(a if b is None else b))
    out    = scores.reshape(tuple(d for s in shapes if len(s) >= 2 for d in s[:1]))
    return out if out.ndim else float(out)


def cosine_scores(a: Any, b: Optional[Any] = None) -> np.ndarray:
    """Pairwise cosine similarity between embeddings; any-dimension inputs are flattened to (n_samples, n_features).

    Args:
        a (Any): List or array: 1D (n,) is one sample; 2D (n_a, n) is a batch; ND keeps axis 0 and flattens the rest.
        b (Optional[Any]): Same forms as `a`; defaults to `a` (self-similarity).

    Returns:
        np.ndarray: Similarity matrix of shape (n_a, n_b), values in [-1, 1].

    Examples:
        >>> query  = [0.1, 0.8, 0.3, 0.5]                                # 1D: one query embedding
        >>> corpus = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]        # 2D: two document embeddings
        >>> scores = cosine_scores(query, corpus)
        >>> scores
        array([[0.981, 0.586]])
        >>> score_doc_0, score_doc_1 = scores[0]                  # unpack the single query row
        >>> cosine_scores(corpus)                                        # self-similarity, shape (2, 2)
        array([[1.   , 0.677],
               [0.677, 1.   ]])
        >>> cosine_scores(np.ones((2, 2, 2)), np.ones((3, 2, 2))).shape  # 3D: trailing axes flattened
        (2, 3)
    """
    return cosine_similarity(_to_matrix(a), None if b is None else _to_matrix(b))
