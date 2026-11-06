from __future__ import annotations
import  numpy as np
from    scipy.optimize            import linear_sum_assignment
from    sklearn                   import metrics
from    sklearn.metrics.cluster   import contingency_matrix
from    sklearn.metrics.pairwise  import cosine_similarity
from    typing                    import Any, Dict, Optional, Union


def _to_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(1, -1) if arr.ndim < 2 else arr.reshape(arr.shape[0], -1)


def _from_matrix(scores: np.ndarray, a: Any, b: Optional[Any] = None) -> Union[float, np.ndarray]:
    """Reshape a `metric_cosine_sim` matrix back to the sample axes of its inputs; 1D inputs drop their axis.

    Args:
        scores (np.ndarray): Similarity matrix of shape (n_a, n_b) from `metric_cosine_sim(a, b)`.
        a (Any): The original first input passed to `metric_cosine_sim`.
        b (Optional[Any]): The original second input; defaults to `a`.

    Returns:
        Union[float, np.ndarray]: Shape `(*a_samples, *b_samples)` — a plain `float` for 1D/1D, (n_b,) for 1D/2D, (n_a, n_b) for 2D/2D.

    Examples:
        >>> query  = [0.1, 0.8, 0.3, 0.5]
        >>> corpus = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]
        >>> _from_matrix(metric_cosine_sim(query, corpus), query, corpus)
        array([0.981, 0.586])
        >>> _from_matrix(metric_cosine_sim(query, query), query, query)
        1.0
    """
    shapes = (np.shape(a), np.shape(a if b is None else b))
    out    = scores.reshape(tuple(d for s in shapes if len(s) >= 2 for d in s[:1]))
    return out if out.ndim else float(out)


def metric_cosine_sim(a: Any, b: Optional[Any] = None) -> np.ndarray:
    """Pairwise cosine similarity between embeddings; any-dimension inputs are flattened to (n_samples, n_features).

    Args:
        a (Any): List or array: 1D (n,) is one sample; 2D (n_a, n) is a batch; ND keeps axis 0 and flattens the rest.
        b (Optional[Any]): Same forms as `a`; defaults to `a` (self-similarity).

    Returns:
        np.ndarray: Similarity matrix of shape (n_a, n_b), values in [-1, 1].

    Examples:
        >>> query  = [0.1, 0.8, 0.3, 0.5]                                    # 1D: one query embedding
        >>> corpus = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]            # 2D: two document embeddings
        >>> scores = metric_cosine_sim(query, corpus)
        >>> scores
        array([[0.981, 0.586]])
        >>> score_doc_0, score_doc_1 = scores[0]                             # unpack the single query row
        >>> metric_cosine_sim(corpus)                                        # self-similarity, shape (2, 2)
        array([[1.   , 0.677],
               [0.677, 1.   ]])
        >>> metric_cosine_sim(np.ones((2, 2, 2)), np.ones((3, 2, 2))).shape  # 3D: trailing axes flattened
        (2, 3)
    """
    return cosine_similarity(_to_matrix(a), None if b is None else _to_matrix(b))


def scorer_clustering(y: Any, yhat: Any) -> Dict[str, Dict[str, float]]:
    """Agreement between known categories and discovered clusters, grouped by what each family of scores measures.

    Args:
        y (Any): Known category per row.
        yhat (Any): Cluster label per row; HDBSCAN noise (`-1`) is scored as one more cluster.

    Returns:
        Dict[str, Dict[str, float]]: Scores keyed by group, then metric; 1.0 is a perfect match except `noise_fraction`.
            `pairwise`: `adjusted_rand` (chance-corrected), `fowlkes_mallows`.
            `information`: `adjusted_mutual_info` (chance-corrected), `normalized_mutual_info`.
            `entropy`: `homogeneity` (one category per cluster), `completeness` (one cluster per category),
                `v_measure` (their harmonic mean); read the first two together.
            `matching`: `purity` (rows in their cluster's majority category), `matched_accuracy` (rows correct under the
                best one-to-one cluster-to-category assignment).
            `outliers`: `noise_fraction` (share of rows labeled `-1`; 0.0 for KMeans).
            Scores that are not chance-corrected rise as clusters multiply.

    Examples:
        >>> labels = HDBSCANDriver(min_cluster_size=2).fit_predict(reduced_faq_embeddings)
        >>> scores = scorer_clustering(df_faq["category"], labels)
        >>> scores["pairwise"]["adjusted_rand"], scores["outliers"]["noise_fraction"]
    """
    y, yhat    = np.asarray(y), np.asarray(yhat)
    counts     = contingency_matrix(y, yhat)                 # rows: categories, columns: clusters
    rows, cols = linear_sum_assignment(counts, maximize=True)
    return {
        # Do pairs of rows land together (or apart) in both the categories and the clusters?
        "pairwise": {
            "adjusted_rand":          float(metrics.adjusted_rand_score(y, yhat)),
            "fowlkes_mallows":        float(metrics.fowlkes_mallows_score(y, yhat)),
        },
        # How much does knowing the cluster tell you about the category?
        "information": {
            "adjusted_mutual_info":   float(metrics.adjusted_mutual_info_score(y, yhat)),
            "normalized_mutual_info": float(metrics.normalized_mutual_info_score(y, yhat)),
        },
        # Are clusters single-category (homogeneity) and categories single-cluster (completeness)?
        "entropy": {
            "homogeneity":            float(metrics.homogeneity_score(y, yhat)),
            "completeness":           float(metrics.completeness_score(y, yhat)),
            "v_measure":              float(metrics.v_measure_score(y, yhat)),
        },
        # What share of rows is right if each cluster is named after its best category?
        "matching": {
            "purity":                 float(counts.max(axis=0).sum() / counts.sum()),
            "matched_accuracy":       float(counts[rows, cols].sum() / counts.sum()),
        },
        # How many rows were left unclustered as noise?
        "outliers": {
            "noise_fraction":         float((yhat == -1).mean()),
        },
    }
