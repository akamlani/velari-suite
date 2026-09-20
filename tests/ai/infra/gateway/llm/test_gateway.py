"""Tests for velari_ai.infra.gateway.llm.gateway."""


def _build_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.infra.gateway.llm.gateway import LLMGateway

    gateway = LLMGateway()
    return gateway


def test_chat_resolves_adapter_and_returns_gateway_response(monkeypatch):
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.openai.provider import OpenAIAdapter
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    gateway = _build_gateway(monkeypatch)
    model_config = types.ModelConfig(provider="openai", model="gpt-4o-mini")
    adapter = gateway._model_registry.resolve(model_config)
    assert isinstance(adapter, OpenAIAdapter)
    monkeypatch.setattr(adapter._client, "chat", lambda **kwargs: types.CompletionResult(
        completion=types.Completion(
            content="Account ACC-10293 has an outstanding balance of $1,204.50.",
            tool_calls=[], finish_reason="stop",
        ),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.0),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    ))

    request = GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="What's the balance on ACC-10293?")])
    result = gateway.chat(request, model_config=model_config)

    assert result.response.message.content == "Account ACC-10293 has an outstanding balance of $1,204.50."
    assert result.metrics.usage.total_tokens == 15


def test_middleware_chain_with_no_middleware_is_pass_through(monkeypatch):
    from velari_ai.infra.gateway.llm.middleware.base import MiddlewareChain
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    chain = MiddlewareChain()
    request = GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="hi")])

    assert chain.run_before(request) is request


def test_embedding_adapter_returns_resolved_adapter_for_dataframe_pipeline(monkeypatch):
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.adapters.base import EmbeddingAdapter

    gateway = _build_gateway(monkeypatch)
    adapter = gateway.embedding_adapter(ModelConfig(provider="openai", model="text-embedding-3-small"))

    assert isinstance(adapter, EmbeddingAdapter)
