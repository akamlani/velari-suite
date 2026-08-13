"""Tests for velari_ai.integrations.agno.agent."""

import pytest


def test_build_sets_agent_and_returns_self():
    from velari_ai.integrations.agno.agent import Agent

    agent = Agent()

    result = agent.build(instructions="You are a billing support assistant.")

    assert result is agent
    assert agent._agent is not None


def test_build_forwards_max_tool_calls_to_tool_call_limit():
    from velari_ai.ai.types import AgentConfig
    from velari_ai.integrations.agno.agent import Agent

    agent = Agent(agent_config=AgentConfig(max_tool_calls=3))

    agent.build()

    assert agent._agent.tool_call_limit == 3


def test_build_raises_for_unsupported_provider():
    from velari_ai.ai.types import ModelConfig
    from velari_ai.integrations.agno.agent import Agent

    agent = Agent(model_config=ModelConfig(model="unsupported:some-model"))

    with pytest.raises(RuntimeError):
        agent.build()


def test_run_without_build_raises_runtimeerror():
    from velari_ai.integrations.agno.agent import Agent

    agent = Agent()

    with pytest.raises(RuntimeError):
        agent.run("What's the balance on ACC-10293?")


def test_run_delegates_to_agno_agent():
    from velari_ai.integrations.agno.agent import Agent

    class _StubAgnoAgent:
        def __init__(self):
            self.calls = []

        def run(self, message):
            self.calls.append(message)
            return f"echo: {message}"

    agent = Agent()
    stub = _StubAgnoAgent()
    agent._agent = stub

    result = agent.run("What's the balance on ACC-10293?")

    assert result == "echo: What's the balance on ACC-10293?"
    assert stub.calls == ["What's the balance on ACC-10293?"]


def test_make_tool_wraps_callable_with_name_and_description():
    from velari_ai.integrations.agno.agent import make_tool

    def lookup_account_balance(account_id: str) -> str:
        """Look up the outstanding balance for a billing account."""
        return f"Account {account_id} balance: $1,204.50"

    tool = make_tool(lookup_account_balance)

    assert tool.name == "lookup_account_balance"
    assert tool.description == "Look up the outstanding balance for a billing account."


def test_make_tool_name_override():
    from velari_ai.integrations.agno.agent import make_tool

    def lookup_account_balance(account_id: str) -> str:
        """Look up the outstanding balance for a billing account."""
        return f"Account {account_id} balance: $1,204.50"

    tool = make_tool(lookup_account_balance, name="get_balance")

    assert tool.name == "get_balance"
