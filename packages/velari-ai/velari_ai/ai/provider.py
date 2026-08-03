from __future__ import annotations
import  numpy as np
import  pandas as pd
from    abc        import ABC, abstractmethod
from    omegaconf  import DictConfig
from    pydantic   import BaseModel
from    typing     import Type, TypeVar, List, Union, Optional, overload
# package modules
from    .types import ProviderMode
from    .evals.scoring import euclidean_distance, cosine_distance, cosine_similarity

T = TypeVar("T", bound=BaseModel)


class Provider(ABC):
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key  = api_key
        self._model    = model
        self._base_url = base_url

    @classmethod
    def from_config(cls, cfg: DictConfig, api_key: str) -> Provider:
        return cls(
            api_key  = api_key,
            model    = cfg.model,
            base_url = cfg.get("base_url", "https://api.openai.com/v1"),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @abstractmethod
    def ask(
        self,
        prompt: str,
        system_prompt: str = "",
        mode: ProviderMode = ProviderMode.CHAT,
        response_format: Optional[Type[T]] = None,
        **kwargs,
    ) -> Union[str, T]: ...


class ProviderEmbeddings(ABC):
    """Abstract base for embedding providers with DataFrame pipeline utilities.

    Examples:
        >>> embedder = OpenAIProviderEmbeddings(api_key=..., embedding_model=...)
        >>> df = pd.DataFrame({"text": ["semantic search over documents", "retrieval-augmented generation", "vector similarity ranking"]})
        >>> # 1. embed a text column into an 'emb' column
        >>> df = embedder.embed(df, col="text")
        >>> # 2. rank rows against a search query (str or pre-computed vector)
        >>> df = embedder.score(df, search_query="how does retrieval-augmented generation work?")
        >>> # 3. extract a (n, d) matrix for clustering / UMAP projection
        >>> emb_matrix = embedder.to_matrix(df)
        >>> # 4. assign cluster labels to dataframe based on distance threshold or nearest neighbors
        >>> # 5. transform to lowr dimensional space for visualization (e.g. UMAP, t-SNE) and plot with seaborn / matplotlib
        >>> emb_trsfrm = umap.fit_transform(emb_matrix)
        >>> labels = df["cluster"]
        >>> sns.scatterplot(x=emb_trsfrm[:, 0], y=emb_trsfrm[:, 1], hue=labels, data=df)
        >>> # 6. annotate truncated text for plotting
        >>> for i, t in enumerate(df[['text']].values):
        >>>     plt.annotate(t, (emb_trsfrm[:, 0][i], emb_trsfrm[:, 1][i]))
    """
    @overload
    def get_embeddings(self, texts: str) -> List[float]: ...
    @overload
    def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...
    @abstractmethod
    def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]: ...

    def to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.vstack(df["emb"].tolist())

    def embed(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.assign(emb=lambda df_: self.get_embeddings(df_[col].tolist()))  # pyright: ignore[reportArgumentType]

    def score(self, df: pd.DataFrame, search_query: Union[str, List[float]]) -> pd.DataFrame:
        query_emb: List[float] = self.get_embeddings(search_query) if isinstance(search_query, str) else search_query
        emb_matrix = self.to_matrix(df)
        return (
            df.assign(
                euclidean_distance = euclidean_distance(emb_matrix, query_emb),
                cosine_distance    = cosine_distance(emb_matrix, query_emb),
                cosine_similarity  = cosine_similarity(emb_matrix, query_emb),
            )
            .sort_values("cosine_similarity", ascending=False)
            .reset_index(drop=True)
        )
