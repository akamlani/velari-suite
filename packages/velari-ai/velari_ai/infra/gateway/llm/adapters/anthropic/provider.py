from    typing import Any, AsyncIterator, Iterator
# package modules
from    ..base import ProviderAdapter
from    ...schemas.types import FinishReason
from    ...schemas.requests import GatewayRequest
from    ...schemas.response import GatewayResponse
from    ......ai.types import CompletionResult, ModelConfig, ProviderName
from    ......integrations.anthropic.client import AnthropicClient

_FINISH_REASONS = {
    "end_turn":   FinishReason.STOP,
    "tool_use":   FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
}


class AnthropicAdapter(ProviderAdapter):
    """Maps `GatewayRequest`/`GatewayResponse` to/from `AnthropicClient` — all Anthropic
    request-building and response-parsing (system/message splitting, the forced-tool
    structured-output trick, content-block walking) lives in
    `integrations/anthropic/client.py`; this class only translates field shapes. Not supported
    by `stream()`/`astream()`.
    """
    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(model_config, **kwargs)
        self._client = AnthropicClient(**self._client_kwargs)

    def _to_gateway_response(
        self, result: CompletionResult, default_finish_reason: FinishReason = FinishReason.ERROR,
    ) -> GatewayResponse:
        finish_reason = _FINISH_REASONS.get(result.completion.finish_reason, default_finish_reason)
        return self._build_gateway_response(
            ProviderName.ANTHROPIC, result.completion, result.metrics.usage, finish_reason, result.metrics.perf.latency_sec,
        )

    def chat(self, request: GatewayRequest) -> GatewayResponse:
        result = self._client.chat(
            model=self._model_config.model, messages=self._to_plain_messages(request),
            tools=request.tools, response_model=request.response_model, **request.parameters,
        )
        return self._to_gateway_response(result)

    async def achat(self, request: GatewayRequest) -> GatewayResponse:
        result = await self._client.achat(
            model=self._model_config.model, messages=self._to_plain_messages(request),
            tools=request.tools, response_model=request.response_model, **request.parameters,
        )
        return self._to_gateway_response(result)

    def stream(self, request: GatewayRequest) -> Iterator[GatewayResponse]:
        if request.response_model is not None:
            raise NotImplementedError("structured output is not supported with stream() yet")
        for result in self._client.stream(
            model=self._model_config.model, messages=self._to_plain_messages(request),
            tools=request.tools, **request.parameters,
        ):
            yield self._to_gateway_response(result, FinishReason.STOP)

    async def astream(self, request: GatewayRequest) -> AsyncIterator[GatewayResponse]:
        if request.response_model is not None:
            raise NotImplementedError("structured output is not supported with stream() yet")
        async for result in self._client.astream(
            model=self._model_config.model, messages=self._to_plain_messages(request),
            tools=request.tools, **request.parameters,
        ):
            yield self._to_gateway_response(result, FinishReason.STOP)
