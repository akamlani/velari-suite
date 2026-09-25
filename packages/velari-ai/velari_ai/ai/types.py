
from    enum        import StrEnum, auto
from    dataclasses import dataclass, field
from    typing      import Any, Dict, List, Optional, Tuple
from    pydantic    import BaseModel
# package modules
from    .schemas.response   import PerfMetrics, UsageMetrics
from    velari_core.core    import ConfigBase

### Constants
MAX_ATTEMPTS         = 3     # maximum number of allowable attempts for certain retryable operations
MAX_ITERATIONS       = 2     # retrieve/rewrite retry budget for agentic RAG loops (see retrieval_agent.py)
MAX_REFLECTIONS      = 3     # critique/revise retry budget for the standalone reflection graphs (retrieval_agent.py)
MAX_TURNS            = 5     # max LLM<->tool round-trips within a single agent call (AgentConfig.max_tool_calls)
MAX_SEARCH_RESULTS   = 10    # max results returned per search query (SearchSpec.max_results_k)
RELEVANCE_THRESHOLD  = 0.5   # minimum heuristic relevance score to accept retrieved context as sufficient

class ProviderName(StrEnum):
    OPENAI                  = auto()
    HUGGINGFACE             = auto()
    ANTHROPIC               = auto()
    SENTENCE_TRANSFORMERS   = auto()

class TaskTurnType(StrEnum):
    SIMPLE                  = auto()
    COMPLEX                 = auto()

    @property
    def turn_range(self) -> Tuple[int, int]:
        ranges = {TaskTurnType.SIMPLE: (3, 5), TaskTurnType.COMPLEX: (15, 30)}
        return ranges[self]

class PersistenceBackend(StrEnum):
    MEMORY                  = auto()
    SQLITE                  = auto()

#### Shared Structures across Vendor Integrations
@dataclass
class ProviderMetrics(object):
    """`PerfMetrics` + `UsageMetrics` for one provider call — identical for every vendor client
    and every call kind (chat or embedding), so `OpenAIClient`/`AnthropicClient`/
    `SentenceTransformerClient` all share this one type instead of each declaring their own.
    """
    perf:  PerfMetrics
    usage: UsageMetrics

@dataclass
class Completion(object):
    """A provider chat call's generated completion — shared by every vendor client's
    `CompletionResult`.
    """
    content:       str
    tool_calls:    List[Dict[str, Any]]
    finish_reason: str
    parsed:        Optional[BaseModel] = field(default=None)
    raw:           Optional[Any]       = field(default=None)

@dataclass
class CompletionResult(object):
    """Plain result of one provider chat call — returned by every vendor client's `chat()`/
    `achat()`/`stream()`/`astream()`, never a raw SDK response type.
    """
    completion: Completion
    metrics:    ProviderMetrics

@dataclass
class EmbeddingVectors(object):
    """An embedding call's output vectors — shared by every vendor client's `EmbeddingResult`."""
    embeddings: List[List[float]]
    raw:        Optional[Any] = field(default=None)

@dataclass
class EmbeddingResult(object):
    """Plain result of one provider embedding call — returned by every vendor client's
    `create_embeddings()`/`embed()`, never a raw SDK response type.
    """
    result:  EmbeddingVectors
    metrics: ProviderMetrics


##### Configuration Structures
@dataclass
class ModelConfig(ConfigBase):
    """Model-level settings for an agent — the provider:model string plus provider-specific kwargs."""
    provider:   str         = field(default="openai")                 # default provider is OpenAI
    model:      str         = field(default="openai:gpt-4o-mini")     # typical no specific version tag
    qos:        str         = field(default="default")                # e.g., lighter/faster (heuristic)
    tags:       List[str]   = field(default_factory=list)             # e.g., ["reasoning", "code"]
    parameters: dict        = field(default_factory=dict)             # e.g., temperature, max_tokens, etc...
    stream:     bool        = field(default=False)                    # whether to stream responses from the model

@dataclass
class AgentConfig(ConfigBase):
    """Agent-level settings — identity and tool-loop behavior."""
    name:           Optional[str] = field(default=None)
    max_tool_calls: int           = field(default=MAX_TURNS)
