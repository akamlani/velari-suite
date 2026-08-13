"""Tests for velari_ai.integrations.langchain.llm."""


class _StubBoundLLM:
    """Hand-written fake standing in for a real `bind_tools()`-wrapped chat model."""
    def __init__(self, responses):
        self._responses = iter(responses)

    def invoke(self, messages):
        return next(self._responses)


def _lookup_account_balance_tool():
    from langchain_core.tools import tool

    @tool
    def lookup_account_balance(account_id: str) -> str:
        """Look up the outstanding balance for a billing account."""
        return f"Account {account_id} balance: $1,204.50"

    return lookup_account_balance


def test_toolcallingllm_bind_sets_bound_llm_and_tools_by_name():
    from velari_ai.integrations.langchain.llm import ToolCallingLLM

    tool = _lookup_account_balance_tool()
    tool_llm = ToolCallingLLM(api_key="test-key")

    bound = tool_llm.bind([tool])

    assert bound is tool_llm._bound_llm
    assert tool_llm._tools_by_name == {"lookup_account_balance": tool}


def test_toolcallingllm_query_without_bind_raises_runtimeerror():
    import pytest
    from velari_ai.integrations.langchain.llm import ToolCallingLLM

    tool_llm = ToolCallingLLM(api_key="test-key")

    with pytest.raises(RuntimeError):
        tool_llm.query("What is the balance on ACC-10293?")


def test_toolcallingllm_query_executes_tool_and_returns_final_answer(monkeypatch):
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.llm import ToolCallingLLM

    tool_llm = ToolCallingLLM(api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])
    monkeypatch.setattr(tool_llm, "_bound_llm", _StubBoundLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_account_balance",
                "args": {"account_id": "ACC-10293"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="Account ACC-10293 has an outstanding balance of $1,204.50."),
    ]))

    result = tool_llm.query("What is the balance on ACC-10293?")

    assert isinstance(result.response, AIMessage)
    assert result.response.content == "Account ACC-10293 has an outstanding balance of $1,204.50."
    assert not result.response.tool_calls


def test_toolcallingllm_query_populates_message_stats_from_tool_calling_turns(monkeypatch):
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.llm import ToolCallingLLM

    tool_llm = ToolCallingLLM(api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])
    monkeypatch.setattr(tool_llm, "_bound_llm", _StubBoundLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_account_balance",
                "args": {"account_id": "ACC-10293"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="Account ACC-10293 has an outstanding balance of $1,204.50."),
    ]))

    result = tool_llm.query("What is the balance on ACC-10293?")

    assert result.metrics.message_stats.cnt_assistant == 2
    assert result.metrics.message_stats.cnt_tool_results == 1
    assert result.metrics.message_stats.cnt_tool_request_turns == 1
    assert result.metrics.message_stats.cnt_tool_requests == 1


def test_toolcallingllm_query_with_response_model_returns_parsed_object_after_tool_calls(monkeypatch):
    from langchain_core.messages import AIMessage
    from pydantic import BaseModel
    from velari_ai.integrations.langchain.llm import ToolCallingLLM

    class _ParsedBalance:
        def __init__(self, balance):
            self.balance = balance

    class BalanceSummary(BaseModel):
        pass

    class _StubStructuredModel:
        def __init__(self, result):
            self._result = result
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return self._result

    class _StubModel:
        def __init__(self, structured_result):
            self._structured_result = structured_result
            self.with_structured_output_calls = []

        def with_structured_output(self, schema):
            self.with_structured_output_calls.append(schema)
            return _StubStructuredModel(self._structured_result)

    parsed = _ParsedBalance(balance=1204.50)
    tool_llm = ToolCallingLLM(api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])
    monkeypatch.setattr(tool_llm, "_bound_llm", _StubBoundLLM([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_account_balance",
                "args": {"account_id": "ACC-10293"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="Account ACC-10293 has an outstanding balance of $1,204.50."),
    ]))
    stub_model = _StubModel(parsed)
    monkeypatch.setattr(tool_llm, "_model", stub_model)

    result = tool_llm.query("What is the balance on ACC-10293?", response_model=BalanceSummary)

    assert result.response is parsed
    assert stub_model.with_structured_output_calls == [BalanceSummary]


def test_toolcallingllm_query_exceeding_max_tool_calls_raises_runtimeerror(monkeypatch):
    import pytest
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.llm import ToolCallingLLM
    from velari_ai.ai.types import AgentConfig

    tool_llm = ToolCallingLLM(agent_config=AgentConfig(max_tool_calls=2), api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])
    looping_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "lookup_account_balance",
            "args": {"account_id": "ACC-1"},
            "id": "call_x",
            "type": "tool_call",
        }],
    )

    class _AlwaysLooping:
        def invoke(self, messages):
            return looping_call

    monkeypatch.setattr(tool_llm, "_bound_llm", _AlwaysLooping())

    with pytest.raises(RuntimeError):
        tool_llm.query("loop forever")


def test_llm_run_sends_system_and_human_messages(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from velari_ai.integrations.langchain.llm import LLM

    class _StubModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="Account ACC-10293 has an outstanding balance of $1,204.50.")

    llm = LLM(api_key="test-key")
    stub = _StubModel()
    monkeypatch.setattr(llm, "_model", stub)

    result = llm.run("You are a billing support assistant.", "What's the balance on ACC-10293?")

    assert isinstance(result.response, AIMessage)
    assert result.response.content == "Account ACC-10293 has an outstanding balance of $1,204.50."
    messages = stub.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "You are a billing support assistant."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "What's the balance on ACC-10293?"


def test_llm_run_populates_usage_stats_from_response(monkeypatch):
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.llm import LLM

    class _StubModel:
        def invoke(self, messages):
            return AIMessage(
                content="Account ACC-10293 has an outstanding balance of $1,204.50.",
                usage_metadata={
                    "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                    "output_token_details": {"reasoning": 12},
                },
            )

    llm = LLM(api_key="test-key")
    monkeypatch.setattr(llm, "_model", _StubModel())

    result = llm.run("You are a billing support assistant.", "What's the balance on ACC-10293?")

    assert result.metrics.latency_sec >= 0
    assert result.metrics.usage_stats.input_tokens == 120
    assert result.metrics.usage_stats.output_tokens == 30
    assert result.metrics.usage_stats.reasoning_tokens == 12
    assert result.metrics.message_stats.cnt_assistant == 1
    assert result.metrics.message_stats.cnt_turn_messages == 3


def test_llm_run_with_response_model_returns_parsed_object(monkeypatch):
    from pydantic import BaseModel
    from velari_ai.integrations.langchain.llm import LLM

    class _ParsedTicket:
        def __init__(self, priority):
            self.priority = priority

    class TicketPriority(BaseModel):
        pass

    class _StubStructuredModel:
        def __init__(self, result):
            self._result = result
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return self._result

    class _StubModel:
        def __init__(self, structured_result):
            self._structured_result = structured_result
            self.with_structured_output_calls = []

        def with_structured_output(self, schema):
            self.with_structured_output_calls.append(schema)
            return _StubStructuredModel(self._structured_result)

    parsed = _ParsedTicket(priority="high")
    llm = LLM(api_key="test-key")
    stub_model = _StubModel(parsed)
    monkeypatch.setattr(llm, "_model", stub_model)

    result = llm.run(
        "Classify this support ticket's priority.", "The site is down for all users.",
        response_model=TicketPriority,
    )

    assert result.response is parsed
    assert isinstance(result.response, _ParsedTicket)
    assert result.response.priority == "high"
    assert stub_model.with_structured_output_calls == [TicketPriority]


def test_llm_batch_invokes_model_once_per_message_with_shared_system_prompt(monkeypatch):
    from langchain_core.messages import AIMessage, SystemMessage
    from velari_ai.integrations.langchain.llm import LLM

    class _StubModel:
        def __init__(self):
            self.calls = []

        def batch(self, inputs):
            self.calls.append(inputs)
            return [AIMessage(content="negative"), AIMessage(content="positive")]

    llm = LLM(api_key="test-key")
    stub = _StubModel()
    monkeypatch.setattr(llm, "_model", stub)

    results = llm.batch(
        "Classify the sentiment of this support ticket as positive, neutral, or negative.",
        ["The product arrived damaged and support has ignored me for a week.", "Great experience, thanks!"],
    )

    assert [r.text for r in results] == ["negative", "positive"]
    inputs = stub.calls[0]
    assert len(inputs) == 2
    assert isinstance(inputs[0][0], SystemMessage)
    assert inputs[0][1].content == "The product arrived damaged and support has ignored me for a week."
    assert inputs[1][1].content == "Great experience, thanks!"


def test_llm_batch_gives_every_item_the_same_latency(monkeypatch):
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.llm import LLM

    class _StubModel:
        def batch(self, inputs):
            return [AIMessage(content="negative"), AIMessage(content="positive")]

    llm = LLM(api_key="test-key")
    monkeypatch.setattr(llm, "_model", _StubModel())

    results = llm.batch(
        "Classify the sentiment of this support ticket as positive, neutral, or negative.",
        ["The product arrived damaged and support has ignored me for a week.", "Great experience, thanks!"],
    )

    assert results[0].metrics.latency_sec == results[1].metrics.latency_sec


def test_llm_stream_yields_chunks_from_model_stream(monkeypatch):
    from langchain_core.messages import SystemMessage
    from velari_ai.integrations.langchain.llm import LLM

    class _StubModel:
        def __init__(self):
            self.calls = []

        def stream(self, messages):
            self.calls.append(messages)
            return iter(["chunk1", "chunk2"])

    llm = LLM(api_key="test-key")
    stub = _StubModel()
    monkeypatch.setattr(llm, "_model", stub)

    chunks = list(llm.stream("You are a research assistant.", "What were Q3's key findings?"))

    assert chunks == ["chunk1", "chunk2"]
    messages = stub.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[1].content == "What were Q3's key findings?"
