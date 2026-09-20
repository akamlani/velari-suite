from    dataclasses import dataclass
# package modules
from    ..schemas.requests import GatewayRequest
from    ..schemas.response import GatewayResponse


@dataclass
class ComplianceDisclaimerMiddleware(object):
    """Appends a mandatory regulatory disclaimer to every assistant response — a concrete
    `GatewayMiddleware` demonstrating a policy-enforcement need distinct from what a framework
    like LangChain ships out of the box: LangChain callbacks/chains are opt-in per app, so
    enforcing that *every* response leaving a regulated org (e.g. "not financial/medical
    advice" in finance/healthcare) carries required legal language means trusting every team
    to remember it. Centralizing it at the gateway makes it non-optional, regardless of which
    team or app placed the call. Not registered by default; opt in via
    `MiddlewareChain([ComplianceDisclaimerMiddleware(disclaimer=...)])`.

    Args:
        disclaimer (str): Regulatory text appended to every response message.

    Examples:
        >>> gateway = LLMGateway(
        ...     middleware=MiddlewareChain([ComplianceDisclaimerMiddleware(disclaimer="Not financial advice.")]),
        ... )
        >>> result = gateway.chat(
        ...     GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="Should I buy this stock?")]),
        ...     model_config=ModelConfig(model="openai:gpt-4o-mini"),
        ... )
        >>> result.response.message.content
        "Past performance doesn't guarantee future returns. Not financial advice."
    """
    disclaimer: str

    def before_request(self, request: GatewayRequest) -> GatewayRequest:
        return request

    def after_response(self, response: GatewayResponse) -> GatewayResponse:
        response.response.message.content = f"{response.response.message.content} {self.disclaimer}"
        return response
