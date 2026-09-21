from    dataclasses import dataclass, field
from    typing      import List, Protocol
# package modules
from    ..schemas.requests import GatewayRequest
from    ..schemas.response import GatewayResponse


class GatewayMiddleware(Protocol):
    """Extension seam for cross-cutting gateway behavior — retry, rate-limit, cache, tracing.

    `PolicyMiddleware` and `LoggingMiddleware` ship as concrete implementations so far;
    `MiddlewareChain` runs whatever is registered, a no-op pass-through when the list is empty.

    Examples:
        >>> gateway = LLMGateway(middleware=MiddlewareChain([PolicyMiddleware(rules=[DisclaimerRule(disclaimer="Not financial advice.")])]))
        >>> result = gateway.chat(
        ...     GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="Should I buy this stock?")]),
        ...     model_config=ModelConfig(model="openai:gpt-4o-mini"),
        ... )
    """
    def before_request(self, request: GatewayRequest) -> GatewayRequest: ...
    def after_response(self, response: GatewayResponse) -> GatewayResponse: ...


@dataclass
class MiddlewareChain(object):
    """Runs an ordered list of `GatewayMiddleware` around a `ProviderAdapter` call.

    Examples:
        >>> chain    = MiddlewareChain([PolicyMiddleware(rules=[DisclaimerRule(disclaimer="Not financial advice.")])])
        >>> request  = chain.run_before(request)
        >>> response = adapter.chat(request)
        >>> response = chain.run_after(response)
    """
    middleware: List[GatewayMiddleware] = field(default_factory=list)

    def run_before(self, request: GatewayRequest) -> GatewayRequest:
        for mw in self.middleware:
            request = mw.before_request(request)
        return request

    def run_after(self, response: GatewayResponse) -> GatewayResponse:
        for mw in self.middleware:
            response = mw.after_response(response)
        return response
