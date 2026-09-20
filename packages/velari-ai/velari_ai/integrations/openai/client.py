from __future__ import annotations
import  json
from    typing    import Any, AsyncIterator, Dict, Iterator, List, Optional, Type, Union, overload
from    omegaconf import DictConfig
from    pydantic  import BaseModel
# specific modules
from    openai import OpenAI, AsyncOpenAI
from    velari_core.core.perf.profiler import profile_time
# package modules
from    ...ai.provider import Provider, ProviderEmbeddings
from    ...ai import types
from    ...ai.schemas.response import PerfMetrics, UsageMetrics


class OpenAIClient(Provider[types.CompletionResult]):
    """Owns everything specific to the OpenAI chat-completions/embeddings SDKs — request
    building, the sync/async/streaming calls themselves, and parsing the raw SDK response
    into a plain `types.CompletionResult`/`types.EmbeddingResult`. Callers (gateway adapters,
    `OpenAIProviderEmbeddings`) never see `openai`'s response types.
    """
    _sync_cls  = OpenAI
    _async_cls = AsyncOpenAI

    @property
    def base_url(self) -> Any:
        return self._client.base_url

    def _to_create_kwargs(self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]], **parameters: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = list(tools)
        return {**kwargs, **parameters}

    def _to_chat_result(self, completion: Any) -> types.CompletionResult:
        choice  = completion.choices[0]
        message = choice.message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
            for tc in (message.tool_calls or [])
        ]
        return types.CompletionResult(
            completion=types.Completion(
                content=message.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "",
                parsed=getattr(message, "parsed", None),
                raw=completion,
            ),
            metrics=types.ProviderMetrics(
                perf=PerfMetrics(latency_sec=0.0),
                usage=UsageMetrics(
                    input_tokens=completion.usage.prompt_tokens,
                    output_tokens=completion.usage.completion_tokens,
                    total_tokens=completion.usage.total_tokens,
                ),
            ),
        )

    def _to_chat_result_chunk(self, chunk: Any) -> types.CompletionResult:
        delta = chunk.choices[0].delta
        return types.CompletionResult(
            completion=types.Completion(
                content=delta.content or "",
                tool_calls=[],
                finish_reason=chunk.choices[0].finish_reason or "",
                raw=chunk,
            ),
            metrics=types.ProviderMetrics(perf=PerfMetrics(latency_sec=0.0), usage=UsageMetrics(total_tokens=0)),
        )

    @profile_time(target=lambda result: result.metrics.perf)
    def chat(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None, **parameters: Any,
    ) -> types.CompletionResult:
        kwargs = self._to_create_kwargs(model, messages, tools, **parameters)
        completion = (
            self._client.chat.completions.parse(**kwargs, response_format=response_model)
            if response_model is not None
            else self._client.chat.completions.create(**kwargs)
        )
        return self._to_chat_result(completion)

    @profile_time(target=lambda result: result.metrics.perf)
    async def achat(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None, **parameters: Any,
    ) -> types.CompletionResult:
        kwargs = self._to_create_kwargs(model, messages, tools, **parameters)
        completion = (
            await self._async_client.chat.completions.parse(**kwargs, response_format=response_model)
            if response_model is not None
            else await self._async_client.chat.completions.create(**kwargs)
        )
        return self._to_chat_result(completion)

    @profile_time(target=lambda result: result.metrics.perf)
    def stream(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **parameters: Any,
    ) -> Iterator[types.CompletionResult]:
        kwargs = self._to_create_kwargs(model, messages, tools, **parameters)
        for chunk in self._client.chat.completions.create(**kwargs, stream=True):
            yield self._to_chat_result_chunk(chunk)

    @profile_time(target=lambda result: result.metrics.perf)
    async def astream(
        self, model: str, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **parameters: Any,
    ) -> AsyncIterator[types.CompletionResult]:
        kwargs = self._to_create_kwargs(model, messages, tools, **parameters)
        stream = await self._async_client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream:
            yield self._to_chat_result_chunk(chunk)

    @profile_time(target=lambda result: result.metrics.perf)
    def create_embeddings(self, model: str, texts: List[str], **parameters: Any) -> types.EmbeddingResult:
        response = self._client.embeddings.create(model=model, input=texts, **parameters)
        return types.EmbeddingResult(
            result=types.EmbeddingVectors(
                embeddings=[data.embedding for data in response.data],
                raw=response,
            ),
            metrics=types.ProviderMetrics(
                perf=PerfMetrics(latency_sec=0.0),
                usage=UsageMetrics(input_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens),
            ),
        )


class OpenAIProviderEmbeddings(ProviderEmbeddings):
    """Embeds via `OpenAIClient.create_embeddings()` — public constructor and `get_embeddings()`
    behavior match the pre-`OpenAIClient` version exactly; only the internal request/response
    handling was consolidated into one place shared with the gateway's `OpenAIEmbeddingAdapter`.
    """
    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._client = OpenAIClient(api_key=api_key, base_url=base_url)
        self._embedding_model = embedding_model

    @classmethod
    def from_config(cls, cfg: DictConfig, api_key: str) -> OpenAIProviderEmbeddings:
        return cls(
            api_key=api_key,
            embedding_model=cfg.embedding,
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
        )

    @overload
    def get_embeddings(self, texts: str) -> List[float]: ...
    @overload
    def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...
    def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        inputs = [texts] if isinstance(texts, str) else texts
        result = self._client.create_embeddings(model=self._embedding_model, texts=inputs)
        return result.result.embeddings[0] if isinstance(texts, str) else result.result.embeddings
