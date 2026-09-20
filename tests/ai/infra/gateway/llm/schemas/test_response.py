"""Tests for velari_ai.infra.gateway.llm.schemas.response."""

import pytest


def test_gateway_embedding_result_embedding_returns_sole_vector():
    from velari_ai.infra.gateway.llm.schemas.response import GatewayEmbeddingResponse

    result = GatewayEmbeddingResponse.Result(embeddings=[[0.1, 0.2, 0.3]])

    assert result.embedding == [0.1, 0.2, 0.3]


def test_gateway_embedding_result_embedding_raises_for_multiple_vectors():
    from velari_ai.infra.gateway.llm.schemas.response import GatewayEmbeddingResponse

    result = GatewayEmbeddingResponse.Result(embeddings=[[0.1], [0.2]])

    with pytest.raises(RuntimeError):
        result.embedding


def test_gateway_response_groups_fields_into_model_metrics_response_scopes():
    from velari_ai.infra.gateway.llm.schemas.types import FinishReason, GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.response import GatewayResponse
    from velari_ai.ai.types import ProviderName
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics

    result = GatewayResponse(
        model=GatewayResponse.Model(provider=ProviderName.OPENAI, model="gpt-4o-mini"),
        metrics=GatewayResponse.Metrics(
            perf=PerfMetrics(latency_sec=0.1),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        response=GatewayResponse.Response(
            message=GatewayMessage(role=Role.ASSISTANT, content="the balance is $42"),
            finish_reason=FinishReason.STOP,
        ),
    )

    assert result.model.provider == ProviderName.OPENAI
    assert result.metrics.usage.total_tokens == 15
    assert result.response.message.content == "the balance is $42"
    assert result.response.finish_reason == FinishReason.STOP
    assert result.response.raw is None
    assert result.response.parsed is None


def test_gateway_response_carries_parsed_structured_output():
    from pydantic import BaseModel
    from velari_ai.ai.types import ProviderName
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.schemas.types import FinishReason, GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.response import GatewayResponse

    class AccountBalance(BaseModel):
        account_id: str
        balance_usd: float

    result = GatewayResponse(
        model=GatewayResponse.Model(provider=ProviderName.OPENAI, model="gpt-4o-mini"),
        metrics=GatewayResponse.Metrics(
            perf=PerfMetrics(latency_sec=0.1),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        response=GatewayResponse.Response(
            message=GatewayMessage(role=Role.ASSISTANT, content=""),
            finish_reason=FinishReason.STOP,
            parsed=AccountBalance(account_id="ACC-10293", balance_usd=1204.50),
        ),
    )

    assert result.response.parsed == AccountBalance(account_id="ACC-10293", balance_usd=1204.50)
