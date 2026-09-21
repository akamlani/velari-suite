import  logging
from    dataclasses import dataclass, field
from    enum        import Flag, auto
from    typing      import Dict
# package modules
from    ..schemas.requests import GatewayRequest
from    ..schemas.response import GatewayResponse

logger = logging.getLogger(__name__)


class LogField(Flag):
    REQUEST  = auto()
    MODEL    = auto()
    FINISH   = auto()
    PERF     = auto()
    USAGE    = auto()
    TOOLS    = auto()
    CONTENT  = auto()
    DEFAULT  = REQUEST | MODEL | FINISH | PERF | USAGE | TOOLS


@dataclass
class LoggingMiddleware(object):
    """Logs the selected `LogField` groups for every request and response passing through the gateway.

    Args:
        level (int): Logging level for emitted records.
        fields (LogField): Groups to log, combined with `|`; unset (or empty) falls back to `LogField.DEFAULT`,
            which omits `CONTENT` since prompts, completions, and tool call arguments may carry sensitive data.
            `TOOLS | CONTENT` adds each tool call's arguments.

    Examples:
        >>> logging_mw = LoggingMiddleware(fields=LogField.MODEL | LogField.USAGE | LogField.TOOLS)
        >>> gateway = LLMGateway(middleware=MiddlewareChain([logging_mw]))
        >>> result = gateway.chat(
        ...     GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="Summarize open tickets for ORG-4471.")]),
        ...     model_config=ModelConfig(model="openai:gpt-4o-mini"),
        ... )
    """
    level:  int       = field(default=logging.INFO)
    fields: LogField  = field(default=LogField.DEFAULT)

    def __post_init__(self) -> None:
        self.fields = self.fields or LogField.DEFAULT

    def _log_summary(self, prefix: str, parts: Dict[LogField, str]) -> None:
        summary = " ".join(text for key, text in parts.items() if key in self.fields)
        if summary:
            logger.log(self.level, f"{prefix}: {summary}")

    def before_request(self, request: GatewayRequest) -> GatewayRequest:
        self._log_summary("Gateway request", {
            LogField.REQUEST: f"messages={len(request.messages)} stream={request.stream} structured={request.response_model is not None}",
            LogField.TOOLS:   f"tools={len(request.tools or [])}",
        })
        if LogField.CONTENT in self.fields:
            logger.log(self.level, f"Gateway request content: {request.messages[-1].content!r}")
        return request

    def after_response(self, response: GatewayResponse) -> GatewayResponse:
        message    = response.response.message
        tool_calls = [
            f"{call.name}({call.arguments})" if LogField.CONTENT in self.fields else call.name
            for call in message.tool_calls or []
        ]
        self._log_summary("Gateway response", {
            LogField.MODEL:  f"provider={response.model.provider} model={response.model.model}",
            LogField.FINISH: f"finish_reason={response.response.finish_reason}",
            LogField.PERF:   f"latency_sec={response.metrics.perf.latency_sec}",
            LogField.USAGE:  f"tokens={response.metrics.usage.total_tokens}",
            LogField.TOOLS:  f"tool_calls={tool_calls}",
        })
        if LogField.CONTENT in self.fields:
            logger.log(self.level, f"Gateway response content: {message.content!r}")
        return response
