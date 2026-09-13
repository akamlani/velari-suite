from __future__ import annotations

import  pandas as pd
from    abc import ABC, abstractmethod
from    typing import List, Optional, Self, Any, Sequence, Tuple, Union
# package modules
from    ..types import ProviderName
from    .types import MetricType, RetrieverSearchType, RetrieverStrategy, SearchScoreResult


### Interface Class for VectorStore
class VectorStore(ABC):
    """Common interface every vector store implementation must satisfy."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def load(self) -> Self: ...

    @abstractmethod
    def create_index(
        self,
        documents: list,
        batch_size: int = 64,
        text_field:      Optional[str] = "page_content",
        embedding_field: Optional[str] = "embedding",
        **kwargs
    ) -> None: ...


    @abstractmethod
    def retrieve_candidates(
        self,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.VECTORSTORE_SIMILARITY,
        k: int = 3,
        **kwargs: Any,
    ) -> Sequence[Union[Any, Tuple[Any, float]]]: ...

    @abstractmethod
    def to_frame(
        self,
        candidates: Sequence[Union[Any, Tuple[Any, float]]],
    ) -> pd.DataFrame: ...

    @abstractmethod
    def to_search_results(
        self,
        candidates: Sequence[Union[Any, Tuple[Any, float]]],
        strategy: RetrieverStrategy = RetrieverStrategy.VECTORSTORE_SIMILARITY,
    ) -> List[SearchScoreResult]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def upsert_merge(
        self,
        texts:      List[str],
        metadatas:  List[dict],
        embeddings: List[List[float]],
        batch_size: int,
    ) -> None: ...

    @abstractmethod
    def exists(self) -> bool: ...

    # @abstractmethod
    # def upsert(
    #     self,
    #     df: pd.DataFrame,
    #     uid: str,
    #     provider: ProviderName,
    #     model: str,
    #     text_col: Optional[str] = None,
    # ) -> int: ...

    # @abstractmethod
    # def search(
    #     self, uid: str, provider: ProviderName, model: str, query_vector: List[float], top_k: int
    # ) -> pd.DataFrame: ...
