"""Tests for velari_ai.integrations.langchain.agent."""

import asyncio

import pytest


def _lookup_account_balance_tool():
    from langchain_core.tools import tool

    @tool
    def lookup_account_balance(account_id: str) -> str:
        """Look up the outstanding balance for a billing account."""
        return f"Account {account_id} balance: $1,204.50"

    return lookup_account_balance


def test_build_sets_agent_and_returns_self():
    from velari_ai.integrations.langchain.agent import Agent

    agent = Agent(api_key="test-key")

    result = agent.build([_lookup_account_balance_tool()], system_prompt="You are a billing support assistant.")

    assert result is agent
    assert "tools" in agent._agent.get_graph().draw_mermaid()


def test_build_uses_inmemory_checkpointer_by_default():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph.state import CompiledStateGraph
    from velari_ai.integrations.langchain.agent import Agent

    agent = Agent(api_key="test-key")

    agent.build([_lookup_account_balance_tool()])

    assert isinstance(agent._agent, CompiledStateGraph)
    assert isinstance(agent._agent.checkpointer, InMemorySaver)


def test_build_forwards_agent_name_to_create_agent():
    from velari_ai.integrations.langchain.agent import Agent
    from velari_ai.ai.types import AgentConfig

    agent = Agent(agent_config=AgentConfig(name="billing-support-agent"), api_key="test-key")

    agent.build([_lookup_account_balance_tool()])

    assert agent._agent.name == "billing-support-agent"


def test_build_forwards_context_schema_to_create_agent(monkeypatch):
    import velari_ai.integrations.langchain.agent as agent_module
    from velari_ai.integrations.langchain.agent import Agent

    captured = {}

    def _fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", _fake_create_agent)
    agent = Agent(api_key="test-key")

    class _DummyContext:
        pass

    agent.build([_lookup_account_balance_tool()], context_schema=_DummyContext)

    assert captured["context_schema"] is _DummyContext


def test_build_forwards_state_schema_to_create_agent(monkeypatch):
    import velari_ai.integrations.langchain.agent as agent_module
    from velari_ai.integrations.langchain.agent import Agent

    captured = {}

    def _fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", _fake_create_agent)
    agent = Agent(api_key="test-key")

    class _DummyState:
        pass

    agent.build([_lookup_account_balance_tool()], state_schema=_DummyState)

    assert captured["state_schema"] is _DummyState


def test_build_defaults_to_sensible_middleware_stack(monkeypatch):
    import velari_ai.integrations.langchain.agent as agent_module
    from langchain.agents.middleware import (
        ModelRetryMiddleware,
        SummarizationMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
    )
    from velari_ai.integrations.langchain.agent import Agent
    from velari_ai.ai.types import AgentConfig

    captured = {}

    def _fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", _fake_create_agent)
    agent = Agent(agent_config=AgentConfig(max_tool_calls=7), api_key="test-key")

    agent.build([_lookup_account_balance_tool()])

    middleware = captured["middleware"]
    assert [type(m) for m in middleware] == [
        ModelRetryMiddleware, ToolRetryMiddleware, ToolCallLimitMiddleware, SummarizationMiddleware,
    ]
    tool_call_limit = middleware[2]
    assert tool_call_limit.run_limit == 7


def test_build_middleware_override_replaces_defaults(monkeypatch):
    import velari_ai.integrations.langchain.agent as agent_module
    from velari_ai.integrations.langchain.agent import Agent

    captured = {}

    def _fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", _fake_create_agent)
    agent = Agent(api_key="test-key")

    agent.build([_lookup_account_balance_tool()], middleware=[])

    assert captured["middleware"] == []


def test_run_without_build_raises_runtimeerror():
    from velari_ai.integrations.langchain.agent import Agent

    agent = Agent(api_key="test-key")

    with pytest.raises(RuntimeError):
        agent.run("What is the balance on ACC-10293?")


def test_run_passes_thread_id_in_config_and_returns_final_message():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        def get_state(self, config):
            class _Snapshot:
                values = {"messages": []}
            return _Snapshot()

        def invoke(self, state, **kwargs):
            self.calls.append((state, kwargs))
            return {"messages": [AIMessage(content="Account ACC-10293 has an outstanding balance of $1,204.50.")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    response = agent.run("What is the balance on ACC-10293?", thread_id="acc-10293-session")

    assert response.response.content == "Account ACC-10293 has an outstanding balance of $1,204.50."
    state, kwargs = stub.calls[0]
    assert state["messages"][0].content == "What is the balance on ACC-10293?"
    assert kwargs["config"]["configurable"]["thread_id"] == "acc-10293-session"


def test_run_without_thread_id_generates_a_fresh_one_each_call():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        def invoke(self, state, **kwargs):
            self.calls.append(kwargs)
            return {"messages": [AIMessage(content="ok")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    agent.run("hi")
    agent.run("hi again")

    # a checkpointer is always attached, so a thread_id is always required — but
    # omitting it still means "no history persists": each call gets its own fresh id.
    first_thread_id = stub.calls[0]["config"]["configurable"]["thread_id"]
    second_thread_id = stub.calls[1]["config"]["configurable"]["thread_id"]
    assert first_thread_id != second_thread_id
    assert stub.calls[0]["context"] is None


def test_run_forwards_context_to_invoke():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent
    from velari_ai.integrations.langchain.types import ContextSchema

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        def invoke(self, state, **kwargs):
            self.calls.append(kwargs)
            return {"messages": [AIMessage(content="ok")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub
    ctx = ContextSchema(experiment_name="test-exp", seed=42)

    agent.run("hi", context=ctx)

    assert stub.calls[0]["context"] is ctx


def test_run_returns_response_info_with_latency():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent
    from velari_ai.integrations.langchain.types import AgentResponseInfo

    class _StubCompiledGraph:
        def invoke(self, state, **kwargs):
            return {"messages": [AIMessage(content="ok")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    agent._agent = _StubCompiledGraph()

    result = agent.run("hi")

    assert isinstance(result, AgentResponseInfo)
    assert result.response.content == "ok"
    assert result.metrics.latency_sec >= 0


def test_run_latency_is_not_accumulated_across_calls():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def invoke(self, state, **kwargs):
            return {"messages": [AIMessage(content="ok")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    agent._agent = _StubCompiledGraph()

    first = agent.run("hi")
    second = agent.run("hi again")

    # each call reports its own latency independently — nothing carried over on self
    assert not hasattr(agent, "stats")
    assert isinstance(first.metrics.latency_sec, float)
    assert isinstance(second.metrics.latency_sec, float)


def test_run_computes_message_stats_from_tool_calling_turns():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def invoke(self, state, **kwargs):
            return {"messages": [
                HumanMessage(content="What is the balance on ACC-1?"),
                AIMessage(content="", tool_calls=[{
                    "name": "lookup_account_balance",
                    "args": {"account_id": "ACC-1"},
                    "id": "call_1",
                    "type": "tool_call",
                }]),
                ToolMessage(content="Account ACC-1 balance: $10", tool_call_id="call_1"),
                AIMessage(content="Your balance is $10."),
            ]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    agent._agent = _StubCompiledGraph()

    result = agent.run("What is the balance on ACC-1?")

    assert result.metrics.message_stats.cnt_assistant == 2
    assert result.metrics.message_stats.cnt_tool_results == 1
    assert result.metrics.message_stats.cnt_tool_request_turns == 1
    assert result.metrics.message_stats.cnt_tool_requests == 1


def test_run_computes_usage_stats_from_ai_message_usage_metadata():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def invoke(self, state, **kwargs):
            return {"messages": [AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                    "output_token_details": {"reasoning": 12},
                },
            )]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    agent._agent = _StubCompiledGraph()

    result = agent.run("hi")

    assert result.metrics.usage_stats.input_tokens == 120
    assert result.metrics.usage_stats.output_tokens == 30
    assert result.metrics.usage_stats.reasoning_tokens == 12


def test_run_message_stats_excludes_prior_thread_history():
    from langchain_core.messages import AIMessage, HumanMessage
    from velari_ai.integrations.langchain.agent import Agent

    prior_messages = [HumanMessage(content="earlier"), AIMessage(content="earlier reply")]

    class _StubCompiledGraph:
        def get_state(self, config):
            class _Snapshot:
                values = {"messages": prior_messages}
            return _Snapshot()

        def invoke(self, state, **kwargs):
            return {"messages": prior_messages + [HumanMessage(content="hi"), AIMessage(content="new reply")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    agent._agent = _StubCompiledGraph()

    result = agent.run("hi", thread_id="thread-1")

    # only this call's new assistant message counts — prior thread history is excluded
    assert result.metrics.message_stats.cnt_assistant == 1
    assert result.metrics.message_stats.cnt_turn_messages == 2
    # ...but cnt_total_messages reports the full thread, including prior history
    assert result.metrics.message_stats.cnt_total_messages == 4
    # ...and ResponseInfo.messages carries that same full thread through for log(history=True)
    assert len(result.messages) == 4


def test_stream_without_build_raises_runtimeerror():
    from velari_ai.integrations.langchain.agent import Agent

    agent = Agent(api_key="test-key")

    with pytest.raises(RuntimeError):
        agent.stream("hi")


def test_stream_yields_chunks_and_passes_thread_id_and_stream_mode():
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        def stream(self, state, **kwargs):
            self.calls.append((state, kwargs))
            return iter([("chunk1", {}), ("chunk2", {})])

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    chunks = list(agent.stream("hi", thread_id="thread-1"))

    assert chunks == [("chunk1", {}), ("chunk2", {})]
    state, kwargs = stub.calls[0]
    assert state["messages"][0].content == "hi"
    assert kwargs["config"]["configurable"]["thread_id"] == "thread-1"
    assert kwargs["stream_mode"] == "messages"


def test_stream_without_thread_id_generates_a_fresh_one():
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        def stream(self, state, **kwargs):
            self.calls.append(kwargs)
            return iter([])

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    list(agent.stream("hi"))

    # a checkpointer is always attached, so a thread_id is always required even
    # when the caller doesn't provide one.
    assert stub.calls[0]["config"]["configurable"]["thread_id"]


def test_arun_without_build_raises_runtimeerror():
    from velari_ai.integrations.langchain.agent import Agent

    agent = Agent(api_key="test-key")

    async def _body():
        await agent.arun("What is the balance on ACC-10293?")

    with pytest.raises(RuntimeError):
        asyncio.run(_body())


def test_arun_passes_thread_id_and_returns_final_message():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        async def aget_state(self, config):
            class _Snapshot:
                values = {"messages": []}
            return _Snapshot()

        async def ainvoke(self, state, **kwargs):
            self.calls.append((state, kwargs))
            return {"messages": [AIMessage(content="Account ACC-10293 has an outstanding balance of $1,204.50.")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    async def _body():
        return await agent.arun("What is the balance on ACC-10293?", thread_id="acc-10293-session")

    response = asyncio.run(_body())

    assert response.response.content == "Account ACC-10293 has an outstanding balance of $1,204.50."
    state, kwargs = stub.calls[0]
    assert state["messages"][0].content == "What is the balance on ACC-10293?"
    assert kwargs["config"]["configurable"]["thread_id"] == "acc-10293-session"


def test_arun_without_thread_id_generates_a_fresh_one_each_call():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, state, **kwargs):
            self.calls.append(kwargs)
            return {"messages": [AIMessage(content="ok")]}

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    async def _body():
        await agent.arun("hi")
        await agent.arun("hi again")

    asyncio.run(_body())

    first_thread_id = stub.calls[0]["config"]["configurable"]["thread_id"]
    second_thread_id = stub.calls[1]["config"]["configurable"]["thread_id"]
    assert first_thread_id != second_thread_id
    assert stub.calls[0]["context"] is None


def test_astream_without_build_raises_runtimeerror():
    from velari_ai.integrations.langchain.agent import Agent

    agent = Agent(api_key="test-key")

    async def _body():
        await agent.astream("hi")

    with pytest.raises(RuntimeError):
        asyncio.run(_body())


def test_astream_yields_chunks_and_passes_thread_id_and_stream_mode():
    from velari_ai.integrations.langchain.agent import Agent

    class _StubCompiledGraph:
        def __init__(self):
            self.calls = []

        async def astream(self, state, **kwargs):
            self.calls.append((state, kwargs))
            yield ("chunk1", {})
            yield ("chunk2", {})

    agent = Agent(api_key="test-key")
    agent.build([_lookup_account_balance_tool()])
    stub = _StubCompiledGraph()
    agent._agent = stub

    async def _body():
        return [chunk async for chunk in await agent.astream("hi", thread_id="thread-1")]

    chunks = asyncio.run(_body())

    assert chunks == [("chunk1", {}), ("chunk2", {})]
    state, kwargs = stub.calls[0]
    assert state["messages"][0].content == "hi"
    assert kwargs["config"]["configurable"]["thread_id"] == "thread-1"
    assert kwargs["stream_mode"] == "messages"


def test_build_wraps_create_agent_errors_in_runtimeerror(monkeypatch):
    import velari_ai.integrations.langchain.agent as agent_module
    from velari_ai.integrations.langchain.agent import Agent

    def _raise(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(agent_module, "create_agent", _raise)
    agent = Agent(api_key="test-key")

    with pytest.raises(RuntimeError):
        agent.build([_lookup_account_balance_tool()])

