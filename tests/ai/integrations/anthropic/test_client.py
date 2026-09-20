"""Tests for velari_ai.integrations.anthropic.client."""


class _FakeAnthropicTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeAnthropicToolUseBlock:
    def __init__(self, id, name, input):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class _FakeAnthropicUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeAnthropicResponse:
    def __init__(self, content, stop_reason, input_tokens=10, output_tokens=5):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _FakeAnthropicUsage(input_tokens, output_tokens)


def test_anthropic_client_chat_splits_system_message_and_sets_max_tokens_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from velari_ai.integrations.anthropic.client import AnthropicClient

    client = AnthropicClient()
    received = {}

    def _fake_create(**kwargs):
        received.update(kwargs)
        return _FakeAnthropicResponse(content=[_FakeAnthropicTextBlock("hi")], stop_reason="end_turn")

    monkeypatch.setattr(client._client.messages, "create", _fake_create)

    client.chat(
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": "You are a billing support assistant."},
            {"role": "user", "content": "What's the balance on ACC-10293?"},
        ],
    )

    assert received["system"] == "You are a billing support assistant."
    assert received["messages"] == [{"role": "user", "content": "What's the balance on ACC-10293?"}]
    assert received["max_tokens"] == 1024


def test_anthropic_client_chat_translates_tool_calls_and_finish_reason(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from velari_ai.integrations.anthropic.client import AnthropicClient

    client = AnthropicClient()
    response = _FakeAnthropicResponse(
        content=[
            _FakeAnthropicTextBlock("Looking that up for you."),
            _FakeAnthropicToolUseBlock(id="toolu_1", name="lookup_account_balance", input={"account_id": "ACC-10293"}),
        ],
        stop_reason="tool_use",
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kwargs: response)

    result = client.chat(model="claude-sonnet-5", messages=[{"role": "user", "content": "What's the balance?"}])

    assert result.completion.finish_reason == "tool_use"
    assert result.completion.content == "Looking that up for you."
    assert result.completion.tool_calls == [{"id": "toolu_1", "name": "lookup_account_balance", "arguments": {"account_id": "ACC-10293"}}]


def test_anthropic_client_chat_carries_parsed_structured_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from pydantic import BaseModel
    from velari_ai.integrations.anthropic.client import AnthropicClient, _STRUCTURED_OUTPUT_TOOL_NAME

    class AccountBalance(BaseModel):
        account_id: str
        balance_usd: float

    client = AnthropicClient()
    received = {}

    def _fake_create(**kwargs):
        received.update(kwargs)
        return _FakeAnthropicResponse(
            content=[_FakeAnthropicToolUseBlock(
                id="toolu_1", name=_STRUCTURED_OUTPUT_TOOL_NAME,
                input={"account_id": "ACC-10293", "balance_usd": 1204.50},
            )],
            stop_reason="tool_use",
        )

    monkeypatch.setattr(client._client.messages, "create", _fake_create)

    result = client.chat(
        model="claude-sonnet-5", messages=[{"role": "user", "content": "What's the balance?"}], response_model=AccountBalance,
    )

    assert received["tool_choice"] == {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL_NAME}
    assert result.completion.parsed == AccountBalance(account_id="ACC-10293", balance_usd=1204.50)
    assert result.completion.tool_calls == []
