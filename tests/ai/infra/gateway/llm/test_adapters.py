"""Tests for velari_ai.infra.gateway.llm.adapters."""

import pytest


def test_openai_adapter_to_gateway_response_maps_tool_calls_and_finish_reason(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.openai.provider import OpenAIAdapter
    from velari_ai.infra.gateway.llm.schemas.types import FinishReason

    adapter = OpenAIAdapter(types.ModelConfig(provider="openai", model="gpt-4o-mini"))
    result = types.CompletionResult(
        completion=types.Completion(
            content="", tool_calls=[{"id": "call_1", "name": "lookup_account_balance", "arguments": {"account_id": "ACC-10293"}}],
            finish_reason="tool_calls",
        ),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.05),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    )

    response = adapter._to_gateway_response(result)

    assert response.response.finish_reason == FinishReason.TOOL_CALLS
    assert response.response.message.tool_calls is not None
    assert response.response.message.tool_calls[0].arguments == {"account_id": "ACC-10293"}
    assert response.metrics.usage.total_tokens == 15


def test_openai_adapter_to_gateway_response_carries_parsed_structured_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from pydantic import BaseModel
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.openai.provider import OpenAIAdapter

    class AccountBalance(BaseModel):
        account_id: str
        balance_usd: float

    adapter = OpenAIAdapter(types.ModelConfig(provider="openai", model="gpt-4o-mini"))
    result = types.CompletionResult(
        completion=types.Completion(
            content="{...}", tool_calls=[], finish_reason="stop",
            parsed=AccountBalance(account_id="ACC-10293", balance_usd=1204.50),
        ),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.0),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    )

    response = adapter._to_gateway_response(result)

    assert response.response.parsed == AccountBalance(account_id="ACC-10293", balance_usd=1204.50)


def test_openai_adapter_stream_raises_when_response_model_requested(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from pydantic import BaseModel
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.adapters.openai.provider import OpenAIAdapter
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    class AccountBalance(BaseModel):
        account_id: str

    adapter = OpenAIAdapter(ModelConfig(provider="openai", model="gpt-4o-mini"))
    request = GatewayRequest(
        messages=[GatewayMessage(role=Role.USER, content="hi")],
        response_model=AccountBalance,
    )

    with pytest.raises(NotImplementedError):
        list(adapter.stream(request))


def test_anthropic_adapter_to_plain_messages_maps_role_and_content(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.adapters.anthropic.provider import AnthropicAdapter
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    adapter = AnthropicAdapter(ModelConfig(provider="anthropic", model="claude-sonnet-5"))
    request = GatewayRequest(messages=[
        GatewayMessage(role=Role.SYSTEM, content="You are a billing support assistant."),
        GatewayMessage(role=Role.USER, content="What's the balance on ACC-10293?"),
    ])

    messages = adapter._to_plain_messages(request)

    assert messages == [
        {"role": "system", "content": "You are a billing support assistant."},
        {"role": "user", "content": "What's the balance on ACC-10293?"},
    ]


def test_anthropic_adapter_to_gateway_response_maps_tool_calls_and_finish_reason(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.anthropic.provider import AnthropicAdapter
    from velari_ai.infra.gateway.llm.schemas.types import FinishReason

    adapter = AnthropicAdapter(types.ModelConfig(provider="anthropic", model="claude-sonnet-5"))
    result = types.CompletionResult(
        completion=types.Completion(
            content="Looking that up for you.",
            tool_calls=[{"id": "toolu_1", "name": "lookup_account_balance", "arguments": {"account_id": "ACC-10293"}}],
            finish_reason="tool_use",
        ),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.05),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    )

    response = adapter._to_gateway_response(result)

    assert response.response.finish_reason == FinishReason.TOOL_CALLS
    assert response.response.message.content == "Looking that up for you."
    assert response.response.message.tool_calls is not None
    assert response.response.message.tool_calls[0].arguments == {"account_id": "ACC-10293"}
    assert response.metrics.usage.total_tokens == 15


def test_anthropic_adapter_to_gateway_response_carries_parsed_structured_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from pydantic import BaseModel
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.anthropic.provider import AnthropicAdapter

    class AccountBalance(BaseModel):
        account_id: str
        balance_usd: float

    adapter = AnthropicAdapter(types.ModelConfig(provider="anthropic", model="claude-sonnet-5"))
    result = types.CompletionResult(
        completion=types.Completion(
            content="", tool_calls=[], finish_reason="tool_use",
            parsed=AccountBalance(account_id="ACC-10293", balance_usd=1204.50),
        ),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.0),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    )

    response = adapter._to_gateway_response(result)

    assert response.response.parsed == AccountBalance(account_id="ACC-10293", balance_usd=1204.50)
    assert response.response.message.tool_calls is None


def test_anthropic_adapter_stream_raises_when_response_model_requested(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from pydantic import BaseModel
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.adapters.anthropic.provider import AnthropicAdapter
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    class AccountBalance(BaseModel):
        account_id: str

    adapter = AnthropicAdapter(ModelConfig(provider="anthropic", model="claude-sonnet-5"))
    request = GatewayRequest(
        messages=[GatewayMessage(role=Role.USER, content="hi")],
        response_model=AccountBalance,
    )

    with pytest.raises(NotImplementedError):
        list(adapter.stream(request))
