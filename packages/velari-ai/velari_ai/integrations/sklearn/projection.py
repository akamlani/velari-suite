from __future__ import annotations
import  numpy as np
import  pandas as pd
import  umap
from    enum    import StrEnum, auto
from    typing  import Any, Dict, Optional, Self
# package modules
from    .driver import ModelDriver


class ProjectionMode(StrEnum):
    VISUALIZE  = auto()
    TRANSFORM  = auto()


class UMAPProjection(ModelDriver):
    """UMAP wrapper for plotting embeddings (`VISUALIZE`) or reducing them for downstream use (`TRANSFORM`).

    Args:
        mode (ProjectionMode): Picks the `n_components` default only: `VISUALIZE` for plots, `TRANSFORM` for downstream use.
        n_components (Optional[int]): Output dimensions; defaults to 2 for `VISUALIZE` and 10 for `TRANSFORM`.
        n_neighbors (int): Local neighborhood size; clamped to the sample count at fit time.
        seed (int): Random state, for reproducible projections.
        **kwargs (Any): Forwarded to `umap.UMAP` (e.g. `min_dist`); `metric` defaults to `"cosine"`.
    """
    def __init__(
        self,
        mode: ProjectionMode = ProjectionMode.VISUALIZE,
        n_components: Optional[int] = None,
        n_neighbors: int = 15,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed)
        self._mode  = mode
        self._model = umap.UMAP(
            n_components = n_components or (2 if mode == ProjectionMode.VISUALIZE else 10),
            n_neighbors  = n_neighbors,
            random_state = seed,
            **{"metric": "cosine", **kwargs},
        )

    @property
    def mode(self) -> ProjectionMode:
        return self._mode

    def fit(self, X: Any, y: Optional[Any] = None) -> Self:
        """Fit UMAP on `X` and return `self`, so calls chain; the embedding is then read via `transform`."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """Fit UMAP and return the embedding of `X`.

        Args:
            X (Any): Embedding matrix of shape (n_samples, n_features).
            y (Optional[Any]): Labels for supervised UMAP; omit for unsupervised.

        Returns:
            np.ndarray: Projection of shape (n_samples, n_components).

        Examples:
            >>> doc_embeddings = embedder.embed_texts(support_ticket_texts)
            >>> projection = UMAPProjection(mode=ProjectionMode.TRANSFORM, n_components=5)
            >>> reduced = projection.fit_transform(doc_embeddings)
            >>> reduced.shape
            (200, 5)
        """
        X = np.asarray(X)
        # k-NN graph needs n_neighbors < n_samples; spectral init needs n_components + 1 < n_samples.
        self._model.set_params(
            n_neighbors  = min(self._model.n_neighbors,  max(2, len(X) - 1)),
            n_components = min(self._model.n_components, max(1, len(X) - 2)),
        )
        embedding    = np.asarray(self._model.fit_transform(X, y))
        self._fitted = True
        return embedding

    def transform(self, X: Any) -> np.ndarray:
        """Project new points into the fitted space.

        Args:
            X (Any): Embedding matrix of shape (n_samples, n_features), same feature count as the fit data.

        Returns:
            np.ndarray: Projection of shape (n_samples, n_components).

        Raises:
            RuntimeError: If the projection has not been fitted.

        Examples:
            >>> projection = UMAPProjection(mode=ProjectionMode.TRANSFORM)
            >>> projection.fit_transform(train_embeddings)
            >>> new_points = projection.transform(new_query_embeddings)
        """
        self._require_fitted()
        return np.asarray(self._model.transform(np.asarray(X)))

    def to_frame(self, X: Any, labels: Optional[Any] = None) -> pd.DataFrame:
        """Fit and return the projection as a plotting-ready frame.

        Args:
            X (Any): Embedding matrix of shape (n_samples, n_features).
            labels (Optional[Any]): Per-sample labels, added as a `label` column (not used for fitting).

        Returns:
            pd.DataFrame: Columns `umap_0..umap_{k-1}`, plus `label` when given.

        Examples:
            >>> projection = UMAPProjection()
            >>> df_proj = projection.to_frame(chunk_embeddings, labels=chunk_source_documents)
            >>> px.scatter(df_proj, x="umap_0", y="umap_1", color="label")
        """
        embedding = self.fit_transform(X)
        df_proj   = pd.DataFrame(embedding, columns=[f"umap_{i}" for i in range(embedding.shape[1])])
        return df_proj if labels is None else df_proj.assign(label=np.asarray(labels))

    def get_parameters(self) -> Dict[str, Any]:
        """Effective settings after any small-sample clamping."""
        return {
            "mode":         str(self._mode),
            "n_components": self._model.n_components,
            "n_neighbors":  self._model.n_neighbors,
        }
