"""Tests for velari_ai.integrations.openai.client."""


class _FakeOpenAIFunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeOpenAIToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeOpenAIFunctionCall(name, arguments)


class _FakeOpenAIMessage:
    def __init__(self, content, tool_calls=None, parsed=None):
        self.content = content
        self.tool_calls = tool_calls
        self.parsed = parsed


class _FakeOpenAIChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeOpenAICompletion:
    def __init__(self, message, finish_reason, prompt_tokens=10, completion_tokens=5):
        self.choices = [_FakeOpenAIChoice(message, finish_reason)]
        self.usage = _FakeOpenAIUsage(prompt_tokens, completion_tokens)


def test_openai_client_chat_omits_tools_when_not_given(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.integrations.openai.client import OpenAIClient

    client = OpenAIClient()
    received = {}

    def _fake_create(**kwargs):
        received.update(kwargs)
        return _FakeOpenAICompletion(message=_FakeOpenAIMessage(content="hi"), finish_reason="stop")

    monkeypatch.setattr(client._client.chat.completions, "create", _fake_create)

    client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert "tools" not in received
    assert received["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_client_chat_translates_tool_calls_and_finish_reason(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.integrations.openai.client import OpenAIClient

    client = OpenAIClient()
    completion = _FakeOpenAICompletion(
        message=_FakeOpenAIMessage(
            content=None,
            tool_calls=[_FakeOpenAIToolCall(id="call_1", name="lookup_account_balance", arguments='{"account_id": "ACC-10293"}')],
        ),
        finish_reason="tool_calls",
    )
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kwargs: completion)

    result = client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert result.completion.finish_reason == "tool_calls"
    assert result.completion.tool_calls == [{"id": "call_1", "name": "lookup_account_balance", "arguments": {"account_id": "ACC-10293"}}]


def test_openai_client_chat_carries_parsed_structured_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from pydantic import BaseModel
    from velari_ai.integrations.openai.client import OpenAIClient

    class AccountBalance(BaseModel):
        account_id: str
        balance_usd: float

    client = OpenAIClient()
    completion = _FakeOpenAICompletion(
        message=_FakeOpenAIMessage(content="{...}", parsed=AccountBalance(account_id="ACC-10293", balance_usd=1204.50)),
        finish_reason="stop",
    )
    monkeypatch.setattr(client._client.chat.completions, "parse", lambda **kwargs: completion)

    result = client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], response_model=AccountBalance)

    assert result.completion.parsed == AccountBalance(account_id="ACC-10293", balance_usd=1204.50)


class _FakeEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingUsage:
    def __init__(self, prompt_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


class _FakeEmbeddingResponse:
    def __init__(self, texts):
        self.data = [_FakeEmbeddingData([float(len(text)), 1.0]) for text in texts]
        self.usage = _FakeEmbeddingUsage(prompt_tokens=10, total_tokens=10)


def test_openai_client_create_embeddings_returns_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.integrations.openai.client import OpenAIClient

    client = OpenAIClient()
    monkeypatch.setattr(client._client.embeddings, "create", lambda *, model, input: _FakeEmbeddingResponse(input))

    result = client.create_embeddings(model="text-embedding-3-small", texts=["a", "bb", "ccc"])

    assert result.result.embeddings == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert result.metrics.usage.total_tokens == 10
