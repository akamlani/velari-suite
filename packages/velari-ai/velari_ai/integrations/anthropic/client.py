from __future__ import annotations
from    typing    import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Type
from    pydantic  import BaseModel
# specific modules
from    anthropic import Anthropic, AsyncAnthropic
from    velari_core.core.perf.profiler import profile_time
# package modules
from    ...ai.provider import Provider
from    ...ai import types
from    ...ai.schemas.response import PerfMetrics, UsageMetrics

_DEFAULT_MAX_TOKENS = 1024
_STRUCTURED_OUTPUT_TOOL_NAME = "extract_structured_response"


class AnthropicClient(Provider[types.CompletionResult]):
    """Owns everything specific to the Anthropic messages SDK — request building (system/
    message splitting, the `max_tokens` requirement, the forced-single-tool structured-output
    trick), the sync/async/streaming calls themselves, and parsing the raw SDK response into a
    plain `types.CompletionResult`. Callers (gateway adapters) never see `anthropic`'s content
    blocks or response types.
    """
    _sync_cls  = Anthropic
    _async_cls = AsyncAnthropic

    def _split_system_and_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest   = [m for m in messages if m["role"] != "system"]
        return system, rest

    def _structured_output_tool(self, response_model: Type[BaseModel]) -> Dict[str, Any]:
        return {
            "name":         _STRUCTURED_OUTPUT_TOOL_NAME,
            "description":  f"Extract the response as {response_model.__name__}.",
            "input_schema": response_model.model_json_schema(),
        }

    def _to_create_kwargs(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]],
        response_model: Optional[Type[BaseModel]], **parameters: Any,
    ) -> Dict[str, Any]:
        system, rest = self._split_system_and_messages(messages)
        parameters = dict(parameters)
        kwargs: Dict[str, Any] = {
            "model":      model,
            "system":     system,
            "messages":   rest,
            "max_tokens": parameters.pop("max_tokens", _DEFAULT_MAX_TOKENS),
        }
        if response_model is not None:
            kwargs["tools"]       = [self._structured_output_tool(response_model)]
            kwargs["tool_choice"] = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL_NAME}
        elif tools:
            kwargs["tools"] = list(tools)
        return {**kwargs, **parameters}

    def _to_chat_result(self, message: Any, response_model: Optional[Type[BaseModel]]) -> types.CompletionResult:
        text_blocks     = [block.text for block in message.content if block.type == "text"]
        tool_use_blocks = [block for block in message.content if block.type == "tool_use"]

        # A forced tool_choice guarantees the only tool_use block is the structured-output one.
        if response_model is not None:
            parsed     = response_model.model_validate(tool_use_blocks[0].input) if tool_use_blocks else None
            tool_calls: List[Dict[str, Any]] = []
        else:
            parsed = None
            tool_calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input)} for b in tool_use_blocks]

        return types.CompletionResult(
            completion=types.Completion(
                content="".join(text_blocks),
                tool_calls=tool_calls,
                finish_reason=message.stop_reason or "",
                parsed=parsed,
                raw=message,
            ),
            metrics=types.ProviderMetrics(
                perf=PerfMetrics(latency_sec=0.0),
                usage=UsageMetrics(
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                    total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                ),
            ),
        )

    @profile_time(target=lambda result: result.metrics.perf)
    def chat(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None, **parameters: Any,
    ) -> types.CompletionResult:
        kwargs  = self._to_create_kwargs(model, messages, tools, response_model, **parameters)
        message = self._client.messages.create(**kwargs)
        return self._to_chat_result(message, response_model)

    @profile_time(target=lambda result: result.metrics.perf)
    async def achat(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None, **parameters: Any,
    ) -> types.CompletionResult:
        kwargs  = self._to_create_kwargs(model, messages, tools, response_model, **parameters)
        message = await self._async_client.messages.create(**kwargs)
        return self._to_chat_result(message, response_model)

    @profile_time(target=lambda result: result.metrics.perf)
    def stream(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **parameters: Any,
    ) -> Iterator[types.CompletionResult]:
        kwargs = self._to_create_kwargs(model, messages, tools, None, **parameters)
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield types.CompletionResult(
                    completion=types.Completion(content=text, tool_calls=[], finish_reason=""),
                    metrics=types.ProviderMetrics(perf=PerfMetrics(latency_sec=0.0), usage=UsageMetrics(total_tokens=0)),
                )

    @profile_time(target=lambda result: result.metrics.perf)
    async def astream(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **parameters: Any,
    ) -> AsyncIterator[types.CompletionResult]:
        kwargs = self._to_create_kwargs(model, messages, tools, None, **parameters)
        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield types.CompletionResult(
                    completion=types.Completion(content=text, tool_calls=[], finish_reason=""),
                    metrics=types.ProviderMetrics(perf=PerfMetrics(latency_sec=0.0), usage=UsageMetrics(total_tokens=0)),
                )
