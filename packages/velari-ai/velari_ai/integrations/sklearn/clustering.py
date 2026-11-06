from __future__ import annotations
import  hdbscan
import  numpy  as np
import  pandas as pd
from    abc              import abstractmethod
from    sklearn.cluster  import KMeans
from    typing           import Any, Dict, Optional, Self
# package modules
from    .driver          import ModelDriver


class ModelClusteringDriver(ModelDriver):
    """Base class for clustering drivers: adds cluster labels, soft membership, and a plotting frame."""

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> Self:
        """Fit on `X` and return `self`, so calls chain; labels are then read from `to_frame` or `fit_predict`."""
        self.fit_predict(X, y)
        return self

    @abstractmethod
    def fit_predict(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """Fit on `X` of shape (n_samples, n_features) and return one cluster label per row."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new rows to the fitted clusters; one label per row."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Score new rows against each fitted cluster; shape (n_samples, n_clusters)."""

    def to_frame(self, X: np.ndarray, labels: Optional[Any] = None) -> pd.DataFrame:
        """Fit and return the cluster assignment as a plotting-ready frame.

        Args:
            X (np.ndarray): Embedding matrix of shape (n_samples, n_features).
            labels (Optional[Any]): Per-sample labels, added as a `label` column (not used for fitting).

        Returns:
            pd.DataFrame: Column `cluster` (`-1` = noise for HDBSCAN), plus `label` when given; subclasses may add more.

        Examples:
            >>> df_clusters = KMeansDriver(n_clusters=4).to_frame(ticket_embeddings, labels=ticket_queues)
            >>> pd.crosstab(df_clusters["cluster"], df_clusters["label"])
        """
        df_clusters = pd.DataFrame({"cluster": self.fit_predict(X)})
        return df_clusters if labels is None else df_clusters.assign(label=np.asarray(labels))


class KMeansDriver(ModelClusteringDriver):
    """K-means with a fixed cluster count; a known-k baseline for comparing against discovered clusters.

    Args:
        n_clusters (int): Requested cluster count; clamped to half the sample count at fit time.
        seed (int): Random state for the centroid initialization.
        **kwargs (Any): Forwarded to `sklearn.cluster.KMeans` (e.g. `n_init`).
    """
    def __init__(self, n_clusters: int = 3, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(seed)
        self._n_clusters = n_clusters
        self._model      = KMeans(n_clusters=n_clusters, random_state=seed, **kwargs)

    def fit_predict(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """Fit on `X` and return one label per row, in `0..n_clusters-1`.

        Args:
            X (np.ndarray): Embedding matrix of shape (n_samples, n_features).
            y (Optional[Any]): Ignored; present for the shared interface.

        Returns:
            np.ndarray: Cluster label per row, shape (n_samples,).

        Examples:
            >>> driver = KMeansDriver(n_clusters=8)
            >>> labels = driver.fit_predict(faq_embeddings)
        """
        X = np.asarray(X)
        self._model.set_params(n_clusters=min(self._n_clusters, max(1, len(X) // 2)))
        labels       = np.asarray(self._model.fit_predict(X))
        self._fitted = True
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new rows to the nearest centroid; one label per row."""
        self._require_fitted()
        return np.asarray(self._model.predict(np.asarray(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax of the negative centroid distances: a relative closeness score, not a calibrated probability.

        Args:
            X (np.ndarray): New rows of shape (n_samples, n_features).

        Returns:
            np.ndarray: Shape (n_samples, n_clusters); each row sums to 1.
        """
        self._require_fitted()
        neg = -np.asarray(self._model.transform(np.asarray(X)))
        exp = np.exp(neg - neg.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def get_parameters(self) -> Dict[str, Any]:
        """Effective `n_clusters`, plus `inertia` and `n_iter` once fitted."""
        params  = {"n_clusters": self._model.n_clusters}
        inertia = getattr(self._model, "inertia_", None)    # absent until fitted
        n_iter  = getattr(self._model, "n_iter_", None)
        return {**params, "inertia": float(inertia), "n_iter": int(n_iter)} if inertia is not None and n_iter is not None else params


class HDBSCANDriver(ModelClusteringDriver):
    """HDBSCAN density clustering; finds the cluster count itself and labels outliers as noise (`-1`).

    Args:
        min_cluster_size (int): Smallest group counted as a cluster; clamped down to the sample count at fit time.
        seed (int): Accepted for the shared interface; HDBSCAN is deterministic, so it is unused.
        prediction_data (bool): Keep what `predict` / `predict_proba` need; must be set before fitting.
        **kwargs (Any): Forwarded to `hdbscan.HDBSCAN` (e.g. `min_samples`, `metric`).
    """
    def __init__(self, min_cluster_size: int = 2, seed: int = 42, prediction_data: bool = True, **kwargs: Any) -> None:
        super().__init__(seed)
        self._min_cluster_size = min_cluster_size
        self._model            = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=prediction_data, **kwargs)

    @property
    def num_clusters(self) -> int:
        """Number of clusters found, excluding noise; 0 before fitting."""
        return len(set(self._model.labels_) - {-1}) if self._fitted else 0

    def fit_predict(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """Fit on `X` and return one label per row; `-1` marks noise.

        Args:
            X (np.ndarray): Embedding matrix of shape (n_samples, n_features); reduce high-dimensional vectors first (e.g. `UMAPProjection`).
            y (Optional[Any]): Ignored; present for the shared interface.

        Returns:
            np.ndarray: Cluster label per row, shape (n_samples,).

        Examples:
            >>> reduced = UMAPProjection(mode=ProjectionMode.TRANSFORM, n_components=5).fit_transform(review_embeddings)
            >>> labels = HDBSCANDriver(min_cluster_size=3).fit_predict(reduced)
        """
        X = np.asarray(X)
        self._model.set_params(min_cluster_size=min(self._min_cluster_size, max(2, len(X))))
        labels       = np.asarray(self._model.fit_predict(X))
        self._fitted = True
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new rows to the fitted clusters, approximately; `-1` marks noise.

        Args:
            X (np.ndarray): New rows of shape (n_samples, n_features), in the space the model was fitted on.

        Returns:
            np.ndarray: Cluster label per row, shape (n_samples,).

        Examples:
            >>> driver = HDBSCANDriver().fit(reduced)
            >>> driver.predict(reducer.transform(query_embeddings))
        """
        self._require_fitted()
        return np.asarray(hdbscan.approximate_predict(self._model, np.asarray(X))[0])    # (labels, strengths)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft membership of new rows in each cluster.

        Args:
            X (np.ndarray): New rows of shape (n_samples, n_features), in the space the model was fitted on.

        Returns:
            np.ndarray: Shape (n_samples, n_clusters); (n_samples, 0) when the fit found no clusters.
        """
        self._require_fitted()
        X = np.asarray(X)
        return hdbscan.membership_vector(self._model, X) if self.num_clusters else np.zeros((len(X), 0))

    def to_frame(self, X: np.ndarray, labels: Optional[Any] = None) -> pd.DataFrame:
        """Fit and return `cluster`, `probability` (strength of membership) and `outlier_score`, plus `label` when given."""
        return super().to_frame(X, labels).assign(probability=self._model.probabilities_, outlier_score=self._model.outlier_scores_)

    def get_parameters(self) -> Dict[str, Any]:
        """Effective `min_cluster_size`, plus `num_clusters`, `noise_cnt` and `probabilities_mean` once fitted."""
        params = {"min_cluster_size": self._model.min_cluster_size}
        stats  = {
            "num_clusters":       self.num_clusters,
            "noise_cnt":          int((self._model.labels_ == -1).sum()) if self._fitted else 0,
            "probabilities_mean": float(np.mean(self._model.probabilities_)) if self._fitted else 0.0,
        }
        return {**params, **stats} if self._fitted else params
