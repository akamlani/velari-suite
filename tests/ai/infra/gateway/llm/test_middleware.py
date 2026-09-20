"""Tests for velari_ai.infra.gateway.llm.middleware.compliance."""


def test_compliance_disclaimer_middleware_before_request_is_pass_through():
    from velari_ai.infra.gateway.llm.middleware.compliance import ComplianceDisclaimerMiddleware
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    middleware = ComplianceDisclaimerMiddleware(disclaimer="Not financial advice.")
    request = GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="Should I buy this stock?")])

    assert middleware.before_request(request) is request


def test_compliance_disclaimer_middleware_appends_disclaimer_to_response():
    from velari_ai.ai.types import ProviderName
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.middleware.compliance import ComplianceDisclaimerMiddleware
    from velari_ai.infra.gateway.llm.schemas.types import FinishReason, GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.response import GatewayResponse

    middleware = ComplianceDisclaimerMiddleware(disclaimer="Not financial advice.")
    response = GatewayResponse(
        model=GatewayResponse.Model(provider=ProviderName.OPENAI, model="gpt-4o-mini"),
        metrics=GatewayResponse.Metrics(
            perf=PerfMetrics(latency_sec=0.1),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        response=GatewayResponse.Response(
            message=GatewayMessage(role=Role.ASSISTANT, content="Past performance doesn't guarantee future returns."),
            finish_reason=FinishReason.STOP,
        ),
    )

    result = middleware.after_response(response)

    assert result is response
    assert result.response.message.content == "Past performance doesn't guarantee future returns. Not financial advice."
