import  logging
import  re
from    abc         import ABC, abstractmethod
from    dataclasses import dataclass, field, replace
from    enum        import StrEnum, auto
from    typing      import Any, Dict, Generic, Iterator, List, Optional, Set, Self, Tuple, Type, TypeVar, Union
# package modules
from    velari_core.core   import ConfigBase
from    ..schemas.requests import GatewayRequest
from    ..schemas.response import GatewayResponse

logger = logging.getLogger(__name__)

PayloadT = TypeVar("PayloadT")

##### Policy Enforcement Types and Errors
class Enforcement(StrEnum):
    ALLOW           = auto()  # PolicyMiddleware skips the rule entirely — check() isn't run, nothing is logged
    BLOCK           = auto()  # on violation, raises PolicyViolationError — before the provider call for a request rule, before the caller sees the reply for a response rule
    MODIFY          = auto()  # on violation, replaces the request/response with a corrected copy from rule.modify() (e.g. redacted text, stripped tool, appended disclaimer) and sends that on
    SHADOW          = auto()  # on violation, logs a warning only; the request/response goes through unchanged

class RuleType(StrEnum):
    TOPIC_DENY      = auto()  # TopicDenyRule
    SENSITIVE_DATA  = auto()  # SensitiveDataRule
    TOOL_ALLOWLIST  = auto()  # ToolAllowlistRule
    DISCLAIMER      = auto()  # DisclaimerRule

class PolicyViolationError(Exception):
    """Raised by `PolicyMiddleware` when a `BLOCK` rule rejects a request or response.

    Args:
        rule (str): Name of the violated rule.
        reason (str): Why the request or response violated it.
        ruleset (Optional[str]): Name of the containing ruleset; `None` for a rule registered on its own.
    """
    def __init__(self, rule: str, reason: str, ruleset: Optional[str] = None) -> None:
        super().__init__(f"Policy '{f'{ruleset}/{rule}' if ruleset else rule}' violated: {reason}")
        self.ruleset = ruleset
        self.rule   = rule
        self.reason = reason


# PolicyRule:        Abstract base class for policy rules; RequestRule runs before the provider call, ResponseRule after it
# ToolAllowlistRule: Enforces an allowlist of tools (request.tools)
# SensitiveDataRule: Detects or redacts sensitive data (SSN, email, card) in request messages
# TopicDenylistRule: Enforces a denylist of topics (request.topics)
# DisclaimerRule:    Ensures every response ends with a required disclaimer
@dataclass(kw_only=True)
class PolicyRule(ABC, Generic[PayloadT]):
    """A single policy on a request or response: `check()` finds a violation, `modify()` optionally fixes it.

    Args:
        name (str): Identifies the rule in errors and logs; defaults to the class name, for simple unnamed rules.
        tags (Set[str]): Labels for selecting rules via `PolicyMiddleware(tags=...)`, e.g. `{"finance", "pii"}`.
        enforcement (Enforcement): `ALLOW` skips the rule, `BLOCK` raises, `MODIFY` applies `modify()`
            and continues, `SHADOW` only logs; `MODIFY` requires a subclass that implements `modify()`.
    """
    name:         str          = field(default="")
    tags:         Set[str]     = field(default_factory=set)
    enforcement:  Enforcement  = field(default=Enforcement.BLOCK)

    def __post_init__(self) -> None:
        self.name = self.name or type(self).__name__
        if self.enforcement == Enforcement.MODIFY and not self.can_modify:
            raise ValueError(f"{type(self).__name__} cannot use Enforcement.MODIFY")

    @property
    def can_modify(self) -> bool:
        return type(self).modify is not PolicyRule.modify

    @abstractmethod
    def check(self, payload: PayloadT, /) -> Optional[str]:
        """Return the violation reason, or `None` when the request/response complies."""

    def modify(self, payload: PayloadT, /) -> PayloadT:
        raise NotImplementedError


@dataclass(kw_only=True)
class RequestRule(PolicyRule[GatewayRequest]):
    """A `PolicyRule` that `PolicyMiddleware` applies to the request, before the provider call."""


@dataclass(kw_only=True)
class ResponseRule(PolicyRule[GatewayResponse]):
    """A `PolicyRule` that `PolicyMiddleware` applies to the response, after the provider call."""


@dataclass(kw_only=True)
class PolicyRuleSet(object):
    """A named group of `PolicyRule`s that can be enforced, shadowed, or switched off together.

    Args:
        name (str): Identifies the ruleset in errors and logs.
        rules (List[PolicyRule[Any]]): Request and response rules, mixed freely; each runs in its own phase, in order.
        tags (Set[str]): Labels every rule in the group inherits, in addition to its own.
        enforcement (Optional[Enforcement]): Overrides every rule's own `enforcement` when set,
            e.g. `SHADOW` to trial a whole ruleset; `MODIFY` requires every rule to implement `modify()`.

    Examples:
        >>> pii_controls = PolicyRuleSet(
        ...     name="pii-controls",
        ...     rules=[SensitiveDataRule(name="redact-pii", enforcement=Enforcement.MODIFY)],
        ... )
    """
    name:         str
    rules:        List[PolicyRule[Any]]
    tags:         Set[str]               = field(default_factory=set)
    enforcement:  Optional[Enforcement]  = field(default=None)

    def __post_init__(self) -> None:
        unsupported = [rule.name for rule in self.rules if not rule.can_modify]
        if self.enforcement == Enforcement.MODIFY and unsupported:
            raise ValueError(f"Ruleset '{self.name}' cannot use Enforcement.MODIFY; no modify() on: {unsupported}")


@dataclass(kw_only=True)
class ToolAllowlistRule(RequestRule):
    """Rejects (or strips) tools outside an allowlist.

    Args:
        allowed (Set[str]): Tool names the caller may offer to the model.

    Examples:
        >>> rule = ToolAllowlistRule(name="support-tools", allowed={"lookup_account_balance"}, enforcement=Enforcement.MODIFY)
    """
    allowed:  Set[str]

    @staticmethod
    def _tool_name(tool: Dict[str, Any]) -> str:
        return tool.get("name") or tool.get("function", {}).get("name", "")

    def _denied(self, request: GatewayRequest) -> List[str]:
        return [name for name in map(self._tool_name, request.tools or []) if name not in self.allowed]

    def check(self, request: GatewayRequest) -> Optional[str]:
        denied = self._denied(request)
        return f"tools not allowed: {denied}" if denied else None

    def modify(self, request: GatewayRequest) -> GatewayRequest:
        kept = [tool for tool in request.tools or [] if self._tool_name(tool) in self.allowed]
        return request.model_copy(update={"tools": kept or None})


@dataclass(kw_only=True)
class SensitiveDataRule(RequestRule):
    """Detects sensitive data (e.g. SSNs, emails, card numbers) in messages before they leave the org.

    Violation reasons list pattern names only, never the matched values. `MODIFY` replaces each
    match with `[REDACTED:<name>]`.

    Args:
        patterns (Dict[str, str]): Regex per data type; defaults to SSN, email, and card number.

    Examples:
        >>> rule = SensitiveDataRule(name="pii-redaction", enforcement=Enforcement.MODIFY)
    """
    patterns:  Dict[str, str]  = field(default_factory=lambda: {
        "ssn":   r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
        "card":  r"\b(?:\d[ -]?){13,16}\b",
    })

    def check(self, request: GatewayRequest) -> Optional[str]:
        found = [name for name, pattern in self.patterns.items() if any(re.search(pattern, message.content) for message in request.messages)]
        return f"sensitive data detected: {found}" if found else None

    def _redact(self, text: str) -> str:
        for name, pattern in self.patterns.items():
            text = re.sub(pattern, f"[REDACTED:{name}]", text)
        return text

    def modify(self, request: GatewayRequest) -> GatewayRequest:
        return request.model_copy(update={"messages": [message.model_copy(update={"content": self._redact(message.content)}) for message in request.messages]})


@dataclass(kw_only=True)
class TopicDenyRule(RequestRule):
    """Rejects requests whose message text matches a denied pattern; cannot use `MODIFY`.

    Args:
        patterns (List[str]): Case-insensitive regexes searched across every message.

    Examples:
        >>> rule = TopicDenyRule(name="no-stock-picks", patterns=[r"which stock(s)? should i (buy|sell)"])
    """
    patterns:  List[str]

    def check(self, request: GatewayRequest) -> Optional[str]:
        return next(
            (f"matched denied topic {pattern!r}" for pattern in self.patterns
             if any(re.search(pattern, message.content, re.IGNORECASE) for message in request.messages)),
            None,
        )

@dataclass(kw_only=True)
class DisclaimerRule(ResponseRule):
    """Ensures every response ends with a required regulatory disclaimer; `MODIFY` appends it if missing.

    Args:
        disclaimer (str): Regulatory text the response must end with.

    Examples:
        >>> rule = DisclaimerRule(name="not-financial-advice", disclaimer="Not financial advice.")
    """
    disclaimer:   str
    enforcement:  Enforcement  = field(default=Enforcement.MODIFY)

    def check(self, response: GatewayResponse) -> Optional[str]:
        return None if response.response.message.content.endswith(self.disclaimer) else "disclaimer missing"

    def modify(self, response: GatewayResponse) -> GatewayResponse:
        message = response.response.message
        message = message.model_copy(update={"content": f"{message.content} {self.disclaimer}"})
        return response.model_copy(update={"response": response.response.model_copy(update={"message": message})})


_RULE_CLASSES: Dict[RuleType, Type[PolicyRule[Any]]] = {
    RuleType.TOPIC_DENY:      TopicDenyRule,
    RuleType.SENSITIVE_DATA:  SensitiveDataRule,
    RuleType.TOOL_ALLOWLIST:  ToolAllowlistRule,
    RuleType.DISCLAIMER:      DisclaimerRule,
}
_COERCIONS: Dict[str, Any] = {"tags": set, "allowed": set, "enforcement": Enforcement}


def _build_rule(entry: Dict[str, Any]) -> PolicyRule[Any]:
    kwargs    = dict(entry)
    rule_type = kwargs.pop("type", None)
    if rule_type is None:
        raise ValueError(f"Rule entry needs a 'type' ({', '.join(RuleType)}): {entry}")
    kwargs = {key: _COERCIONS[key](value) if key in _COERCIONS else value for key, value in kwargs.items()}
    return _RULE_CLASSES[RuleType(rule_type)](**kwargs)


def _build_item(entry: Dict[str, Any]) -> Union[PolicyRule[Any], PolicyRuleSet]:
    if "ruleset" not in entry:
        return _build_rule(entry)
    return PolicyRuleSet(
        name=entry["ruleset"],
        rules=[_build_rule(rule) for rule in entry.get("rules", [])],
        tags=set(entry.get("tags", [])),
        enforcement=Enforcement(entry["enforcement"]) if entry.get("enforcement") else None,
    )


@dataclass
class PolicyConfig(ConfigBase):
    """Policy settings read from YAML — the rule entries plus how they layer, mirroring `ModelConfig`.

    Args:
        rules (List[Dict[str, Any]]): Rule entries (`type` is a `RuleType`, the rest are that rule's fields)
            and ruleset entries (`ruleset` name, plus its own `rules`, `tags`, `enforcement`).
        tags (List[str]): Only rules with one of these tags (their own or their ruleset's) run; empty runs all.
        file (Optional[str]): Policy YAML to layer on top of this one; resolved by the caller, not by this class.
        extend (bool): For a layered-on config: `True` appends its rules to the base's, `False` replaces them.

    Examples:
        >>> generic  = PolicyConfig.from_config(cfg.policy)
        >>> support  = PolicyConfig.from_config(OmegaConf.load("config/usecases/cs/policy.yaml"))
        >>> policy   = PolicyMiddleware.from_config(generic.merge(support))
    """
    rules:   List[Dict[str, Any]]  = field(default_factory=list)
    tags:    List[str]             = field(default_factory=list)
    file:    Optional[str]         = field(default=None)
    extend:  bool                  = field(default=True)

    def merge(self, overlay: Self) -> Self:
        """Layer `overlay` on top of this config; its `tags` win when set, otherwise this config's are kept."""
        return replace(self, rules=(self.rules if overlay.extend else []) + overlay.rules, tags=overlay.tags or self.tags)


@dataclass
class PolicyMiddleware(object):
    """Enforces policies on every request before the provider call and every response after it.

    Args:
        rules (List[Union[PolicyRule[Any]], PolicyRuleSet]): Standalone rules and rulesets, run in order;
            `RequestRule`s act on the request and `ResponseRule`s on the response. A `BLOCK` violation
            raises `PolicyViolationError`.
        tags (Set[str]): When set, only rules with at least one matching tag (their own or their
            ruleset's) run; empty runs everything.

    Examples:
        >>> policy = PolicyMiddleware(rules=[
        ...     SensitiveDataRule(enforcement=Enforcement.MODIFY),
        ...     PolicyRuleSet(name="advice-controls", tags={"finance"}, rules=[
        ...         TopicDenyRule(name="no-stock-picks", patterns=[r"which stock(s)? should i (buy|sell)"]),
        ...         DisclaimerRule(name="not-financial-advice", disclaimer="Not financial advice."),
        ...     ]),
        ... ])
        >>> gateway = LLMGateway(middleware=MiddlewareChain([policy]))
        >>> gateway.chat(GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="Which stock should I buy?")]))
        Traceback (most recent call last):
        PolicyViolationError: Policy 'advice-controls/no-stock-picks' violated: matched denied topic 'which stock(s)? should i (buy|sell)'
    """
    rules:  List[Union[PolicyRule[Any], PolicyRuleSet]]  = field(default_factory=list)
    tags:   Set[str]                                     = field(default_factory=set)

    @classmethod
    def from_config(cls, entry: Union[PolicyConfig, Dict[str, Any]]) -> Self:
        """Build a policy from a `PolicyConfig`, or a raw mapping (e.g. parsed YAML) converted via `PolicyConfig.from_config()`.

        Examples:
            >>> policy = PolicyMiddleware.from_config({
            ...     "rules": [
            ...         {"type": "sensitive_data", "name": "redact-pii", "tags": ["pii"], "enforcement": "modify"},
            ...         {"ruleset": "advice-controls", "rules": [{"type": "disclaimer", "disclaimer": "Not financial advice."}]},
            ...     ],
            ... })
        """
        config = entry if isinstance(entry, PolicyConfig) else PolicyConfig.from_config(entry)
        return cls(rules=[_build_item(item) for item in config.rules], tags=set(config.tags))

    @property
    def labels(self) -> List[str]:
        """Every rule as `ruleset/rule`, or just `rule` when standalone, in run order."""
        return [f"{ruleset.name}/{rule.name}" if ruleset else rule.name for ruleset, rule in self._entries()]

    def with_enforcement(self, overrides: Dict[str, Enforcement]) -> Self:
        """Return a copy with `enforcement` overridden by rule or ruleset name; raises `ValueError` on an unknown name."""
        known   = {name for ruleset, rule in self._entries() for name in (rule.name, ruleset.name if ruleset else rule.name)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown rule or ruleset name(s) {sorted(unknown)}; known: {sorted(known)}")

        def override(item: Any) -> Any:
            return replace(item, enforcement=overrides[item.name]) if item.name in overrides else item

        return replace(self, rules=[
            override(replace(item, rules=[override(rule) for rule in item.rules])) if isinstance(item, PolicyRuleSet) else override(item)
            for item in self.rules
        ])

    def _entries(self) -> Iterator[Tuple[Optional[PolicyRuleSet], PolicyRule[Any]]]:
        for item in self.rules:
            if isinstance(item, PolicyRuleSet):
                yield from ((item, rule) for rule in item.rules)
            else:
                yield None, item

    def _enforce(self, payload: PayloadT, rule_type: Type[PolicyRule[PayloadT]]) -> PayloadT:
        for ruleset, rule in self._entries():
            tags        = rule.tags | (ruleset.tags if ruleset else set())
            enforcement = (ruleset.enforcement if ruleset else None) or rule.enforcement
            label       = f"{ruleset.name}/{rule.name}" if ruleset else rule.name
            if not isinstance(rule, rule_type) or enforcement == Enforcement.ALLOW or (self.tags and not self.tags & tags):
                continue
            reason = rule.check(payload)
            if reason is None:
                continue
            if enforcement == Enforcement.BLOCK:
                raise PolicyViolationError(rule.name, reason, ruleset.name if ruleset else None)
            level = logging.WARNING if enforcement == Enforcement.SHADOW else logging.INFO
            logger.log(level, f"Policy '{label}' violated ({enforcement}, tags={sorted(tags)}): {reason}")
            if enforcement == Enforcement.MODIFY:
                payload = rule.modify(payload)
        return payload

    def before_request(self, request: GatewayRequest) -> GatewayRequest:
        return self._enforce(request, RequestRule)

    def after_response(self, response: GatewayResponse) -> GatewayResponse:
        return self._enforce(response, ResponseRule)
