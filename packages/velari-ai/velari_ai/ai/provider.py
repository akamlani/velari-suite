from __future__ import annotations
import  numpy as np
import  pandas as pd
from    abc    import ABC, abstractmethod
from    typing import Any, AsyncIterator, ClassVar, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union, overload
from    pydantic import BaseModel
# package modules
from    .evals.scoring import euclidean_distance, cosine_distance, cosine_similarity

TResponseResult = TypeVar("TResponseResult")


class Provider(ABC, Generic[TResponseResult]):
    """Abstract base for vendor chat clients — the sync/async/streaming/structured-output
    request+parse layer beneath `infra/gateway/llm/adapters/<vendor>/provider.py`'s field-mapping.
    Each vendor's client (`OpenAIClient`, `AnthropicClient`) implements this over its own SDK
    and returns the shared `CompletionResult` (`ai/types.py`), never a raw SDK response type —
    mirrors `infra/gateway/llm/adapters/base.py`'s `ProviderAdapter` one layer up.

    Concrete subclasses set `_sync_cls`/`_async_cls` to their vendor's SDK client classes;
    `__init__` builds both from the same kwargs, so subclasses only need to declare them.

    Args:
        **kwargs (Any): Forwarded to both `_sync_cls()` and `_async_cls()` — e.g. `api_key`,
            `base_url`, `timeout`.
    """
    _sync_cls:  ClassVar[Type[Any]]
    _async_cls: ClassVar[Type[Any]]

    def __init__(self, **kwargs: Any) -> None:
        self._client       = self._sync_cls(**kwargs)
        self._async_client = self._async_cls(**kwargs)

    @abstractmethod
    def chat(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None, **parameters: Any,
    ) -> TResponseResult: ...

    @abstractmethod
    async def achat(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None, **parameters: Any,
    ) -> TResponseResult: ...

    @abstractmethod
    def stream(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **parameters: Any,
    ) -> Iterator[TResponseResult]: ...

    @abstractmethod
    def astream(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **parameters: Any,
    ) -> AsyncIterator[TResponseResult]: ...


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
