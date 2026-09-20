from    abc    import ABC, abstractmethod
from    typing import Any, AsyncIterator, Dict, Iterator, List, Union, overload
# package modules
from    ..schemas import types as gateway_types
from    ..schemas import requests as gateway_requests
from    ..schemas import response as gateway_response
from    .....ai.provider import ProviderEmbeddings
from    .....ai import types
from    .....ai.schemas.response import PerfMetrics, UsageMetrics

_ResultResponse = types.Completion
_ResultMetrics  = UsageMetrics

# Remaining known gap: an OpenAI Responses-API request mode (chat-completions only today).
# Structured output (GatewayRequest.response_model) is implemented.


class GatewayAdapterMixin(object):
    """Shared `ModelConfig` -> (`_model_config`, `_client_kwargs`) construction for gateway
    adapters — mixed into `ProviderAdapter` (chat) and `EmbeddingAdapter` (embeddings), two
    otherwise-unrelated interfaces that both start from a `ModelConfig`.

    Args:
        model_config (ModelConfig): Provider:model + parameters this adapter was resolved for.
        **kwargs (Any): Extra provider kwargs (e.g. `api_key`); wins over `model_config.extra`.
    """
    def __init__(self, model_config: types.ModelConfig, **kwargs: Any) -> None:
        self._model_config  = model_config
        self._client_kwargs = {**model_config.extra, **kwargs}

    def _to_plain_messages(self, request: gateway_requests.GatewayRequest) -> List[Dict[str, str]]:
        return [{"role": str(m.role), "content": m.content} for m in request.messages]

    def _build_gateway_response(
        self,
        provider:      types.ProviderName,
        response:      _ResultResponse,
        metrics:       _ResultMetrics,
        finish_reason: gateway_types.FinishReason,
        latency_sec:   float,
    ) -> gateway_response.GatewayResponse:
        tool_calls = [gateway_types.GatewayToolCall(**tc) for tc in response.tool_calls] or None
        return gateway_response.GatewayResponse(
            model=gateway_response.GatewayResponse.Model(provider=provider, model=self._model_config.model),
            metrics=gateway_response.GatewayResponse.Metrics(
                perf=PerfMetrics(latency_sec=round(latency_sec, 3)),
                usage=metrics,
            ),
            response=gateway_response.GatewayResponse.Response(
                message=gateway_types.GatewayMessage(role=gateway_types.Role.ASSISTANT, content=response.content, tool_calls=tool_calls),
                finish_reason=finish_reason,
                raw=response.raw,
                parsed=response.parsed,
            ),
        )


class ProviderAdapter(GatewayAdapterMixin, ABC):
    """One provider's translation between `GatewayRequest`/`GatewayResponse` and its own SDK."""
    @abstractmethod
    def chat(self, request: gateway_requests.GatewayRequest) -> gateway_response.GatewayResponse: ...

    @abstractmethod
    async def achat(self, request: gateway_requests.GatewayRequest) -> gateway_response.GatewayResponse: ...

    @abstractmethod
    def stream(self, request: gateway_requests.GatewayRequest) -> Iterator[gateway_response.GatewayResponse]: ...

    @abstractmethod
    def astream(self, request: gateway_requests.GatewayRequest) -> AsyncIterator[gateway_response.GatewayResponse]: ...


class EmbeddingAdapter(GatewayAdapterMixin, ProviderEmbeddings, ABC):
    """Provider embedding adapter: a unified `GatewayEmbeddingResponse` via `embed_texts()`,
    plus the inherited `embed()`/`score()`/`to_matrix()` DataFrame pipeline from
    `ProviderEmbeddings` (built on this class's `get_embeddings()`, itself implemented once
    here in terms of `embed_texts()` — concrete adapters only need to implement that one method).
    """
    @abstractmethod
    def embed_texts(self, request: gateway_requests.GatewayEmbeddingRequest) -> gateway_response.GatewayEmbeddingResponse: ...

    @overload
    def get_embeddings(self, texts: str) -> List[float]: ...
    @overload
    def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...
    def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        request = gateway_requests.GatewayEmbeddingRequest(texts=[texts] if isinstance(texts, str) else texts)
        vectors = self.embed_texts(request).result.embeddings
        return vectors[0] if isinstance(texts, str) else vectors
