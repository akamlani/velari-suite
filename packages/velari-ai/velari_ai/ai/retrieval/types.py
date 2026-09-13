from __future__ import annotations

from    dataclasses import dataclass, field
from    enum         import StrEnum, auto
from    typing       import Any, Dict, Required, NotRequired, TypedDict
# package modules
from    velari_core.config import ConfigBase
from    ..types import MAX_SEARCH_RESULTS

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

@dataclass(frozen=True)
class SearchSpec:
    # TODO: extend for query with additional filters such as {search terms, phrase, domain, category, published date}
    max_results_k: int = field(default=MAX_SEARCH_RESULTS)  # maximum search results to return for each search query

class SearchScoreResult(TypedDict, total=False):
    text:          Required[Any]            # Document class instance or string
    _distance:     NotRequired[float]
    _relevance:    NotRequired[float]
    _rerank_score: NotRequired[float]

class ChunkingStrategy(StrEnum):
    CHARACTER                   = auto()
    TOKEN                       = auto()
    SENTENCE_TRANSFORMER_TOKEN  = auto()
    RECURSIVE_CHARACTER         = auto()
    MARKDOWN_HEADER             = auto()


@dataclass
class ChunkingConfig(ConfigBase):
    """Chunking strategy + size/overlap + strategy-specific extra kwargs for DocumentChunker.

    Examples:
        >>> cfg = Filesystem.read("examples/ai/agents/_conf/graph_agent.yaml")
        >>> chunking_config = ChunkingConfig.from_config(cfg.retrieval.chunking)
        >>> chunker = DocumentChunker(chunking_config)
    """
    strategy:      ChunkingStrategy = field(default=ChunkingStrategy.RECURSIVE_CHARACTER)
    chunk_size:    int              = field(default=512)
    chunk_overlap: int              = field(default=50)

    @classmethod
    def _coerce_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "strategy" in kwargs:
            kwargs["strategy"] = ChunkingStrategy(kwargs["strategy"])
        return kwargs


@dataclass
class RetrievalConfig(ConfigBase):
    """Retrieval strategy + result count + strategy-specific extra kwargs for VectorStore.retrieve_candidates().

    Examples:
        >>> cfg = Filesystem.read("examples/ai/agents/_conf/graph_agent.yaml")
        >>> retrieval_config = RetrievalConfig.from_config(cfg.retrieval.search)
        >>> retrieve_node = RetrieveNode(vector_store=vector_store, config=retrieval_config)
    """
    strategy: RetrieverStrategy = field(default=RetrieverStrategy.VECTORSTORE_RELEVANCE_SCORE)
    k:        int               = field(default=3)

    @classmethod
    def _coerce_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "strategy" in kwargs:
            kwargs["strategy"] = RetrieverStrategy(kwargs["strategy"])
        return kwargs
