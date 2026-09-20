from    dataclasses import dataclass, field
from    typing      import Any, Dict, Generic, Type, TypeVar
# specific modules
from    velari_data.storage import LocalKeyValueStore
# package modules
from    .adapters.base import EmbeddingAdapter, GatewayAdapterMixin, ProviderAdapter
from    .adapters.openai.provider import OpenAIAdapter
from    .adapters.openai.embedding import OpenAIEmbeddingAdapter
from    .adapters.anthropic.provider import AnthropicAdapter
from    .adapters.stf.embedding import SentenceTransformersEmbeddingAdapter
from    ....ai.types import ModelConfig, ProviderName

_DEFAULT_ADAPTERS: Dict[str, Type[ProviderAdapter]] = {
    str(ProviderName.OPENAI):    OpenAIAdapter,
    str(ProviderName.ANTHROPIC): AnthropicAdapter,
}

_DEFAULT_EMBEDDING_ADAPTERS: Dict[str, Type[EmbeddingAdapter]] = {
    str(ProviderName.OPENAI):                OpenAIEmbeddingAdapter,
    str(ProviderName.HUGGINGFACE):           SentenceTransformersEmbeddingAdapter,
    str(ProviderName.SENTENCE_TRANSFORMERS): SentenceTransformersEmbeddingAdapter,
}

TAdapter = TypeVar("TAdapter", bound=GatewayAdapterMixin)


@dataclass
class _AdapterRegistry(Generic[TAdapter]):
    """Resolves a `ModelConfig` to a cached, provider-specific adapter via `ProviderName`.

    Backed by `velari_data.storage.LocalKeyValueStore` for both the provider->adapter-class
    table and the model cache — reused rather than hand-rolled, since it already gives
    dict-like put/get/has semantics. `velari_data.registry.Registry` itself doesn't fit here:
    it's built around a single static catalog loaded once via `create(cfg)` and read back by
    plain name (see `PromptRegistry`), whereas `resolve()` is a two-tier class lookup plus a
    model cache that grows lazily, one adapter per (provider, model) pair. Shared as a generic
    base by `ModelRegistry` (chat) and `EmbeddingRegistry` (embeddings) so this resolve/cache
    logic isn't duplicated between them.

    Args:
        adapters (LocalKeyValueStore[Type[TAdapter]]): Provider name -> adapter class.
    """
    adapters:     LocalKeyValueStore[Type[TAdapter]]
    _model_cache: LocalKeyValueStore[TAdapter] = field(default_factory=lambda: LocalKeyValueStore(), init=False, repr=False)

    def resolve(self, model_config: ModelConfig, **kwargs: Any) -> TAdapter:
        """Return the cached adapter for `model_config`, constructing it on first use.

        Args:
            model_config (ModelConfig): Selects both the adapter class (via `.provider`) and
                the model/parameters passed to it.
            **kwargs (Any): Extra adapter kwargs (e.g. `api_key`); only used on first construction.

        Raises:
            ValueError: If `model_config.provider` has no registered adapter.
        """
        provider = ProviderName(model_config.provider)
        try:
            adapter_cls = self.adapters[str(provider)]
        except KeyError:
            raise ValueError(f"No adapter registered for provider={provider!r}") from None

        cache_key = f"{provider}:{model_config.model}"
        try:
            return self._model_cache[cache_key]
        except KeyError:
            adapter = adapter_cls(model_config, **kwargs)
            self._model_cache.put(cache_key, adapter)
            return adapter


@dataclass
class ModelRegistry(_AdapterRegistry[ProviderAdapter]):
    """Resolves a `ModelConfig` to a cached, provider-specific `ProviderAdapter` (chat).

    Args:
        adapters (LocalKeyValueStore[Type[ProviderAdapter]]): Provider name -> adapter class;
            defaults to the built-in OpenAI/Anthropic adapters. Extend with more providers by
            passing a `LocalKeyValueStore` seeded with a superset of this default.

    Examples:
        >>> registry = ModelRegistry()
        >>> adapter = registry.resolve(ModelConfig(provider="openai", model="gpt-4o-mini"))
    """
    adapters: LocalKeyValueStore[Type[ProviderAdapter]] = field(default_factory=lambda: LocalKeyValueStore(dict(_DEFAULT_ADAPTERS)))


@dataclass
class EmbeddingRegistry(_AdapterRegistry[EmbeddingAdapter]):
    """Resolves a `ModelConfig` to a cached, provider-specific `EmbeddingAdapter`.

    Args:
        adapters (LocalKeyValueStore[Type[EmbeddingAdapter]]): Provider name -> adapter class;
            defaults to the built-in OpenAI/HuggingFace/SentenceTransformers adapters (the
            latter two synonymous, per `ProviderEmbeddingFactory`). Extend with more providers
            by passing a `LocalKeyValueStore` seeded with a superset of this default.

    Examples:
        >>> registry = EmbeddingRegistry()
        >>> adapter = registry.resolve(ModelConfig(provider="openai", model="text-embedding-3-small"))
    """
    adapters: LocalKeyValueStore[Type[EmbeddingAdapter]] = field(default_factory=lambda: LocalKeyValueStore(dict(_DEFAULT_EMBEDDING_ADAPTERS)))
