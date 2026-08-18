from __future__ import annotations

import  pandas as pd
from    abc import ABC, abstractmethod
from    typing import List, Optional, Self, Any, Sequence, Tuple, Union
from    enum import StrEnum, auto
# package modules
from    ..types import ProviderName

class RetrieverStrategy(StrEnum):
    VECTORSTORE_SIMILARITY       = auto()
    VECTORSTORE_DISTANCE_SCORE   = auto()
    VECTORSTORE_RELEVANCE_SCORE  = auto()
    VECTORSTORE_MMR              = auto()
    VECTORSTORE_RETRIEVER        = auto()

class RetrieverSearchType(StrEnum):
    SIMILARITY              = auto()
    SIMILARITY_THRESHOLD    = auto()
    MMR                     = auto()

class MetricType(StrEnum):
    DISTANCE    = auto()
    RELEVANCE   = auto()


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
