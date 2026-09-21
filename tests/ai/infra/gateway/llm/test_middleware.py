"""Tests for velari_ai.infra.gateway.llm.middleware."""
import logging


def _make_response(content="Past performance doesn't guarantee future returns.", tool_calls=None):
    from velari_ai.ai.types import ProviderName
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.schemas.types import FinishReason, GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.response import GatewayResponse

    return GatewayResponse(
        model=GatewayResponse.Model(provider=ProviderName.OPENAI, model="gpt-4o-mini"),
        metrics=GatewayResponse.Metrics(
            perf=PerfMetrics(latency_sec=0.1),
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        response=GatewayResponse.Response(
            message=GatewayMessage(role=Role.ASSISTANT, content=content, tool_calls=tool_calls),
            finish_reason=FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP,
        ),
    )


class TestDisclaimerRule:
    def test_after_response_appends_disclaimer_without_mutating_original(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import DisclaimerRule, PolicyMiddleware

        policy = PolicyMiddleware(rules=[DisclaimerRule(disclaimer="Not financial advice.")])
        response = _make_response()

        result = policy.after_response(response)

        assert result.response.message.content == "Past performance doesn't guarantee future returns. Not financial advice."
        assert response.response.message.content == "Past performance doesn't guarantee future returns."

    def test_after_response_disclaimer_already_present_passes_through(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import DisclaimerRule, PolicyMiddleware

        policy = PolicyMiddleware(rules=[DisclaimerRule(disclaimer="Not financial advice.")])
        response = _make_response(content="Diversify your holdings. Not financial advice.")

        assert policy.after_response(response) is response

    def test_after_response_block_enforcement_raises_when_disclaimer_missing(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import DisclaimerRule, Enforcement, PolicyMiddleware, PolicyViolationError

        policy = PolicyMiddleware(rules=[DisclaimerRule(disclaimer="Not financial advice.", enforcement=Enforcement.BLOCK)])

        with pytest.raises(PolicyViolationError, match="disclaimer missing"):
            policy.after_response(_make_response())


class TestLoggingMiddleware:
    def test_after_response_default_logs_summary_without_content(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.core import LoggingMiddleware

        with caplog.at_level(logging.INFO):
            LoggingMiddleware().after_response(_make_response(content="secret reply"))

        assert "model=gpt-4o-mini" in caplog.text
        assert "tokens=15" in caplog.text
        assert "secret reply" not in caplog.text

    def test_after_response_content_flag_logs_content(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.core import LogField, LoggingMiddleware

        with caplog.at_level(logging.INFO):
            LoggingMiddleware(fields=LogField.CONTENT).after_response(_make_response(content="secret reply"))

        assert "secret reply" in caplog.text

    def test_after_response_only_logs_selected_fields(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.core import LogField, LoggingMiddleware

        with caplog.at_level(logging.INFO):
            LoggingMiddleware(fields=LogField.USAGE).after_response(_make_response())

        assert "tokens=15" in caplog.text
        assert "model=gpt-4o-mini" not in caplog.text

    def test_empty_fields_falls_back_to_default(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.core import LogField, LoggingMiddleware

        with caplog.at_level(logging.INFO):
            LoggingMiddleware(fields=LogField(0)).after_response(_make_response())

        assert "model=gpt-4o-mini" in caplog.text

    def test_after_response_tool_call_arguments_require_tools_and_content(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.core import LogField, LoggingMiddleware
        from velari_ai.infra.gateway.llm.schemas.types import GatewayToolCall

        tool_calls = [GatewayToolCall(id="call_1", name="lookup_account_balance", arguments={"account_id": "ACC-10293"})]

        with caplog.at_level(logging.INFO):
            LoggingMiddleware(fields=LogField.TOOLS).after_response(_make_response(tool_calls=tool_calls))
        assert "lookup_account_balance" in caplog.text
        assert "ACC-10293" not in caplog.text

        caplog.clear()
        with caplog.at_level(logging.INFO):
            LoggingMiddleware(fields=LogField.TOOLS | LogField.CONTENT).after_response(_make_response(tool_calls=tool_calls))
        assert "ACC-10293" in caplog.text

    def test_before_request_content_flag_logs_last_message(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.core import LogField, LoggingMiddleware
        from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
        from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

        request = GatewayRequest(messages=[GatewayMessage(role=Role.USER, content="Summarize tickets for ORG-4471.")])

        with caplog.at_level(logging.INFO):
            result = LoggingMiddleware(fields=LogField.CONTENT).before_request(request)

        assert result is request
        assert "ORG-4471" in caplog.text


def _policy(*rules, **ruleset_kwargs):
    from velari_ai.infra.gateway.llm.middleware.compliance import PolicyMiddleware, PolicyRuleSet

    return PolicyMiddleware(rules=[PolicyRuleSet(name="test-ruleset", rules=list(rules), **ruleset_kwargs)])


def _user_request(content, **kwargs):
    from velari_ai.infra.gateway.llm.schemas.types import GatewayMessage, Role
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayRequest

    return GatewayRequest(messages=[GatewayMessage(role=Role.USER, content=content)], **kwargs)


class TestPolicyMiddleware:
    def test_before_request_block_rule_raises_with_ruleset_and_rule_name(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyViolationError, TopicDenyRule

        policy = _policy(TopicDenyRule(name="no-stock-picks", patterns=[r"which stock should i buy"]))

        with pytest.raises(PolicyViolationError, match="test-ruleset/no-stock-picks"):
            policy.before_request(_user_request("Which stock should I buy?"))

    def test_before_request_compliant_request_passes_through_unchanged(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import SensitiveDataRule, TopicDenyRule

        policy = _policy(
            TopicDenyRule(name="no-stock-picks", patterns=[r"which stock should i buy"]),
            SensitiveDataRule(name="pii"),
        )
        request = _user_request("What's the balance on ACC-10293?")

        assert policy.before_request(request) is request

    def test_before_request_tool_allowlist_modify_strips_disallowed_tools(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, ToolAllowlistRule

        policy = _policy(ToolAllowlistRule(name="support-tools", allowed={"lookup_account_balance"}, enforcement=Enforcement.MODIFY))
        tools = [{"name": "lookup_account_balance"}, {"type": "function", "function": {"name": "close_account"}}]
        request = _user_request("Close my account.", tools=tools)

        result = policy.before_request(request)

        assert result.tools == [{"name": "lookup_account_balance"}]
        assert request.tools == tools

    def test_before_request_sensitive_data_modify_redacts_without_mutating_original(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, SensitiveDataRule

        policy = _policy(SensitiveDataRule(name="pii", enforcement=Enforcement.MODIFY))
        request = _user_request("My SSN is 123-45-6789, email jo@example.com.")

        result = policy.before_request(request)

        assert result.messages[0].content == "My SSN is [REDACTED:ssn], email [REDACTED:email]."
        assert "123-45-6789" in request.messages[0].content

    def test_before_request_sensitive_data_block_reason_omits_matched_value(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyViolationError, SensitiveDataRule

        policy = _policy(SensitiveDataRule(name="pii"))

        with pytest.raises(PolicyViolationError) as exc_info:
            policy.before_request(_user_request("My SSN is 123-45-6789."))

        assert "ssn" in exc_info.value.reason
        assert "123-45-6789" not in str(exc_info.value)

    def test_before_request_shadow_rule_logs_and_passes_through(self, caplog):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, TopicDenyRule

        policy = _policy(TopicDenyRule(name="no-stock-picks", patterns=[r"which stock should i buy"], enforcement=Enforcement.SHADOW))
        request = _user_request("Which stock should I buy?")

        with caplog.at_level(logging.WARNING):
            result = policy.before_request(request)

        assert result is request
        assert "test-ruleset/no-stock-picks" in caplog.text

    def test_before_request_allow_rule_is_skipped(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, TopicDenyRule

        policy = _policy(TopicDenyRule(name="no-stock-picks", patterns=[r"which stock should i buy"], enforcement=Enforcement.ALLOW))
        request = _user_request("Which stock should I buy?")

        assert policy.before_request(request) is request

    def test_before_request_ruleset_enforcement_overrides_rule_enforcement(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, TopicDenyRule

        policy = _policy(
            TopicDenyRule(name="no-stock-picks", patterns=[r"which stock should i buy"], enforcement=Enforcement.BLOCK),
            enforcement=Enforcement.SHADOW,
        )
        request = _user_request("Which stock should I buy?")

        assert policy.before_request(request) is request

    def test_ruleset_applies_request_and_response_rules_in_their_own_phase(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import DisclaimerRule, TopicDenyRule

        policy = _policy(
            TopicDenyRule(name="no-stock-picks", patterns=[r"which stock should i buy"]),
            DisclaimerRule(name="not-financial-advice", disclaimer="Not financial advice."),
        )
        request = _user_request("What's the balance on ACC-10293?")

        assert policy.before_request(request) is request
        assert policy.after_response(_make_response()).response.message.content.endswith("Not financial advice.")

    def test_before_request_unnamed_rule_defaults_name_to_class_name(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyMiddleware, PolicyViolationError, TopicDenyRule

        policy = PolicyMiddleware(rules=[TopicDenyRule(patterns=[r"which stock should i buy"])])

        with pytest.raises(PolicyViolationError, match="'TopicDenyRule' violated"):
            policy.before_request(_user_request("Which stock should I buy?"))

    def test_before_request_tag_filter_skips_rules_without_matching_tag(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyMiddleware, TopicDenyRule

        policy = PolicyMiddleware(
            rules=[TopicDenyRule(patterns=[r"which stock should i buy"], tags={"finance"})],
            tags={"pii"},
        )
        request = _user_request("Which stock should I buy?")

        assert policy.before_request(request) is request

    def test_before_request_tag_filter_matches_ruleset_tags(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyMiddleware, PolicyRuleSet, PolicyViolationError, SensitiveDataRule

        policy = PolicyMiddleware(
            rules=[PolicyRuleSet(name="pii-controls", tags={"pii"}, rules=[SensitiveDataRule()])],
            tags={"pii"},
        )

        with pytest.raises(PolicyViolationError, match="pii-controls/SensitiveDataRule"):
            policy.before_request(_user_request("My SSN is 123-45-6789."))

    def test_from_config_builds_simple_structured_and_ruleset_entries(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, PolicyMiddleware, PolicyRuleSet, SensitiveDataRule

        policy = PolicyMiddleware.from_config({
            "tags": ["pii"],
            "rules": [
                {"type": "topic_deny", "patterns": ["guaranteed returns?"]},
                {"type": "sensitive_data", "name": "redact-pii", "tags": ["pii"], "enforcement": "modify"},
                {"ruleset": "advice-controls", "tags": ["finance"], "rules": [{"type": "disclaimer", "name": "notice", "disclaimer": "Not financial advice."}]},
            ],
        })

        assert policy.labels == ["TopicDenyRule", "redact-pii", "advice-controls/notice"]
        assert policy.tags == {"pii"}
        assert isinstance(policy.rules[1], SensitiveDataRule)
        assert (policy.rules[1].tags, policy.rules[1].enforcement) == ({"pii"}, Enforcement.MODIFY)
        assert isinstance(policy.rules[2], PolicyRuleSet)

    def test_from_config_unknown_rule_type_raises_valueerror(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyMiddleware

        with pytest.raises(ValueError):
            PolicyMiddleware.from_config({"rules": [{"type": "nonexistent"}]})

    def test_with_enforcement_overrides_by_ruleset_and_rule_name(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, PolicyMiddleware, PolicyRuleSet

        policy = PolicyMiddleware.from_config({
            "rules": [
                {"type": "sensitive_data", "name": "redact-pii", "enforcement": "modify"},
                {"ruleset": "advice-controls", "rules": [{"type": "topic_deny", "name": "no-stock-picks", "patterns": ["stock"]}]},
            ],
        })

        result = policy.with_enforcement({"redact-pii": Enforcement.SHADOW, "advice-controls": Enforcement.ALLOW})

        ruleset = result.rules[1]
        assert isinstance(ruleset, PolicyRuleSet)
        assert (result.rules[0].enforcement, ruleset.enforcement) == (Enforcement.SHADOW, Enforcement.ALLOW)
        assert policy.rules[0].enforcement == Enforcement.MODIFY

    def test_with_enforcement_unknown_name_raises_valueerror(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, PolicyMiddleware

        policy = PolicyMiddleware.from_config({"rules": [{"type": "topic_deny", "name": "no-stock-picks", "patterns": ["stock"]}]})

        with pytest.raises(ValueError, match="nope"):
            policy.with_enforcement({"nope": Enforcement.SHADOW})

    def test_topic_deny_rule_modify_enforcement_raises_valueerror(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, TopicDenyRule

        with pytest.raises(ValueError):
            TopicDenyRule(name="no-stock-picks", patterns=[r"stock"], enforcement=Enforcement.MODIFY)

    def test_ruleset_modify_enforcement_with_unmodifiable_rule_raises_valueerror(self):
        import pytest
        from velari_ai.infra.gateway.llm.middleware.compliance import Enforcement, PolicyRuleSet, TopicDenyRule

        with pytest.raises(ValueError, match="no-stock-picks"):
            PolicyRuleSet(
                name="advice-controls",
                rules=[TopicDenyRule(name="no-stock-picks", patterns=[r"stock"])],
                enforcement=Enforcement.MODIFY,
            )


class TestPolicyConfig:
    def test_from_config_reads_declared_fields_and_keeps_unknown_in_extra(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyConfig

        config = PolicyConfig.from_config({"rules": [{"type": "topic_deny", "patterns": ["stock"]}], "tags": ["finance"], "owner": "support-team"})

        assert (config.tags, config.extend, config.extra) == (["finance"], True, {"owner": "support-team"})
        assert config.rules == [{"type": "topic_deny", "patterns": ["stock"]}]

    def test_merge_extend_appends_overlay_rules(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyConfig

        base = PolicyConfig(rules=[{"type": "sensitive_data", "name": "redact-pii"}], tags=["pii"])
        overlay = PolicyConfig(rules=[{"type": "topic_deny", "name": "no-refund-promises", "patterns": ["refund"]}])

        merged = base.merge(overlay)

        assert [rule["name"] for rule in merged.rules] == ["redact-pii", "no-refund-promises"]
        assert merged.tags == ["pii"]

    def test_merge_extend_false_replaces_base_rules_and_overlay_tags_win(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyConfig

        base = PolicyConfig(rules=[{"type": "sensitive_data", "name": "redact-pii"}], tags=["pii"])
        overlay = PolicyConfig(rules=[{"type": "disclaimer", "name": "notice", "disclaimer": "Not advice."}], tags=["finance"], extend=False)

        merged = base.merge(overlay)

        assert [rule["name"] for rule in merged.rules] == ["notice"]
        assert merged.tags == ["finance"]

    def test_policy_middleware_from_config_accepts_policy_config(self):
        from velari_ai.infra.gateway.llm.middleware.compliance import PolicyConfig, PolicyMiddleware

        config = PolicyConfig(rules=[{"type": "sensitive_data", "name": "redact-pii"}], tags=["pii"])

        policy = PolicyMiddleware.from_config(config)

        assert (policy.labels, policy.tags) == (["redact-pii"], {"pii"})
