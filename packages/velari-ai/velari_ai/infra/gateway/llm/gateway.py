from    typing import Any, AsyncIterator, Iterator, List, Optional
# package modules
from    .adapters.base import EmbeddingAdapter
from    .middleware.base import MiddlewareChain
from    .registry import EmbeddingRegistry, ModelRegistry
from    .schemas import requests, response
from    ....ai.types import ModelConfig


class LLMGateway(object):
    """Single entry point for provider-agnostic chat and embedding calls — resolves a
    provider, runs chat calls through the middleware chain, and returns a unified
    `GatewayResponse`/`GatewayEmbeddingResponse`.

    Args:
        registry (Optional[ModelRegistry]): Provider -> chat adapter resolution; defaults to
            `ModelRegistry()` (built-in OpenAI/Anthropic adapters).
        embedding_registry (Optional[EmbeddingRegistry]): Provider -> embedding adapter
            resolution; defaults to `EmbeddingRegistry()` (built-in OpenAI/HuggingFace/
            SentenceTransformers adapters).
        middleware (Optional[MiddlewareChain]): Cross-cutting hooks run around every chat
            call; defaults to an empty, no-op `MiddlewareChain()`. Not applied to `embed()`
            in this iteration.

    Examples:
        >>> gateway = LLMGateway()
        >>> request = GatewayRequest(messages=[
        ...     GatewayMessage(role=Role.SYSTEM, content="You are a billing support assistant."),
        ...     GatewayMessage(role=Role.USER, content="What's the balance on ACC-10293?"),
        ... ])
        >>> result = gateway.chat(request, model_config=ModelConfig(provider="openai", model="gpt-4o-mini"))
        >>> result.response.message.content
        'Account ACC-10293 has an outstanding balance of $1,204.50.'
    """
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        embedding_registry: Optional[EmbeddingRegistry] = None,
        middleware: Optional[MiddlewareChain] = None,
    ) -> None:
        self._model_registry     = registry or ModelRegistry()
        self._embedding_registry = embedding_registry or EmbeddingRegistry()
        self._middleware         = middleware or MiddlewareChain()

    def chat(self, request: requests.GatewayRequest, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> response.GatewayResponse:
        model_config = model_config or ModelConfig()
        adapter  = self._model_registry.resolve(model_config, **kwargs)
        request  = self._middleware.run_before(request)
        result   = adapter.chat(request)
        return self._middleware.run_after(result)

    async def achat(self, request: requests.GatewayRequest, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> response.GatewayResponse:
        model_config = model_config or ModelConfig()
        adapter  = self._model_registry.resolve(model_config, **kwargs)
        request  = self._middleware.run_before(request)
        result   = await adapter.achat(request)
        return self._middleware.run_after(result)

    def stream(self, request: requests.GatewayRequest, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> Iterator[response.GatewayResponse]:
        model_config = model_config or ModelConfig()
        adapter = self._model_registry.resolve(model_config, **kwargs)
        request = self._middleware.run_before(request)
        for result in adapter.stream(request):
            yield self._middleware.run_after(result)

    async def astream(self, request: requests.GatewayRequest, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> AsyncIterator[response.GatewayResponse]:
        model_config = model_config or ModelConfig()
        adapter = self._model_registry.resolve(model_config, **kwargs)
        request = self._middleware.run_before(request)
        async for result in adapter.astream(request):
            yield self._middleware.run_after(result)

    def embed(self, texts: List[str], model_config: Optional[ModelConfig] = None, **kwargs: Any) -> response.GatewayEmbeddingResponse:
        """Embed a batch of texts, resolving the provider from `model_config`.

        Examples:
            >>> gateway = LLMGateway()
            >>> result = gateway.embed(
            ...     ["retrieval-augmented generation", "vector similarity search"],
            ...     model_config=ModelConfig(provider="openai", model="text-embedding-3-small"),
            ... )
            >>> len(result.result.embeddings)
            2
        """
        adapter = self.embedding_adapter(model_config, **kwargs)
        return adapter.embed_texts(requests.GatewayEmbeddingRequest(texts=texts))

    def embedding_adapter(self, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> EmbeddingAdapter:
        """Resolve and return the `EmbeddingAdapter` itself, for callers who want the
        inherited `ProviderEmbeddings` DataFrame pipeline (`.embed(df, col)`, `.score(df,
        search_query)`, `.to_matrix(df)`) rather than `embed()`'s single-call wrapper.

        Examples:
            >>> adapter = LLMGateway().embedding_adapter(ModelConfig(provider="openai", model="text-embedding-3-small"))
            >>> df = adapter.embed(df, col="text")
        """
        model_config = model_config or ModelConfig()
        return self._embedding_registry.resolve(model_config, **kwargs)
