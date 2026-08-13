"""Tests for velari_ai.integrations.langchain.utils."""


def _lookup_account_balance_tool():
    from langchain_core.tools import tool

    @tool
    def lookup_account_balance(account_id: str) -> str:
        """Look up the outstanding balance for a billing account."""
        return f"Account {account_id} balance: $1,204.50"

    return lookup_account_balance


def test_render_graph_none_context_raises_runtimeerror():
    import pytest
    from velari_ai.integrations.langchain.utils import render_graph

    with pytest.raises(RuntimeError):
        render_graph(None)


def test_render_graph_default_returns_mermaid():
    from velari_ai.integrations.langchain.llm import ToolCallingLLM
    from velari_ai.integrations.langchain.utils import render_graph

    tool_llm = ToolCallingLLM(api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])

    result = render_graph(tool_llm._bound_llm)

    assert "graph TD" in result
    assert "ChatOpenAI" in result


def test_render_graph_use_ascii_returns_ascii_art(monkeypatch):
    from langchain_core.runnables.graph import Graph
    from velari_ai.integrations.langchain.llm import ToolCallingLLM
    from velari_ai.integrations.langchain.utils import render_graph

    monkeypatch.setattr(Graph, "draw_ascii", lambda self: "+-----------+\n| __start__ |\n+-----------+")
    tool_llm = ToolCallingLLM(api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])

    result = render_graph(tool_llm._bound_llm, use_ascii=True)

    assert result == "+-----------+\n| __start__ |\n+-----------+"


def test_render_graph_with_path_writes_png(monkeypatch, tmp_path):
    from langchain_core.runnables.graph import Graph
    from velari_ai.integrations.langchain.llm import ToolCallingLLM
    from velari_ai.integrations.langchain.utils import render_graph

    written_paths = []

    def _fake_draw_mermaid_png(self, *, output_file_path: str, **kwargs):
        written_paths.append(output_file_path)
        with open(output_file_path, "wb") as f:
            f.write(b"fake-png-bytes")
        return b"fake-png-bytes"

    monkeypatch.setattr(Graph, "draw_mermaid_png", _fake_draw_mermaid_png)
    tool_llm = ToolCallingLLM(api_key="test-key")
    tool_llm.bind([_lookup_account_balance_tool()])
    png_path = tmp_path / "nested" / "agent_graph.png"

    render_graph(tool_llm._bound_llm, path=str(png_path))

    assert written_paths == [str(png_path)]
    assert png_path.exists()


def test_mcp_tools_to_langchain_converts_each_tool(monkeypatch):
    import velari_ai.integrations.langchain.utils as utils_module
    from velari_ai.integrations.langchain.utils import mcp_tools_to_langchain

    captured = []

    def _fake_convert(*, session, tool, connection, server_name):
        captured.append((tool, connection, server_name))
        return object()

    monkeypatch.setattr(utils_module, "convert_mcp_tool_to_langchain_tool", _fake_convert)

    class _FakeTool:
        name = "get_time"

    connection = {"transport": "stdio", "command": "python", "args": ["server.py"]}
    result = mcp_tools_to_langchain([_FakeTool()], connection, server_name="search")

    assert len(result) == 1
    tool, conn, server_name = captured[0]
    assert tool.name == "get_time"
    assert conn == connection
    assert server_name == "search"


def test_log_response_writes_pretty_printed_response(caplog):
    import logging
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.types import MessageStats, Metrics, ResponseInfo, UsageStats
    from velari_ai.integrations.langchain.utils import log_response

    metrics = Metrics(
        latency_sec=0.5,
        usage_stats=UsageStats(input_tokens=10, output_tokens=5, reasoning_tokens=0),
        message_stats=MessageStats(cnt_total_messages=1, cnt_turn_messages=1, cnt_assistant=1),
    )
    response = AIMessage(content="Account ACC-1 balance: $10")
    info = ResponseInfo(response=response, metrics=metrics, messages=[response])

    with caplog.at_level(logging.INFO, logger="velari_ai.integrations.langchain.utils"):
        log_response(info)

    assert "Account ACC-1 balance: $10" in caplog.text
    assert "Ai Message" in caplog.text


def test_log_response_history_logs_every_message(caplog):
    import logging
    from langchain_core.messages import AIMessage, HumanMessage
    from velari_ai.integrations.langchain.types import MessageStats, Metrics, ResponseInfo, UsageStats
    from velari_ai.integrations.langchain.utils import log_response

    messages = [
        HumanMessage(content="earlier question"),
        AIMessage(content="earlier reply"),
        HumanMessage(content="hi"),
        AIMessage(content="new reply"),
    ]
    metrics = Metrics(
        latency_sec=0.5,
        usage_stats=UsageStats(input_tokens=10, output_tokens=5, reasoning_tokens=0),
        message_stats=MessageStats(cnt_total_messages=4, cnt_turn_messages=2, cnt_assistant=1),
    )
    info = ResponseInfo(response=messages[-1], metrics=metrics, messages=messages)

    with caplog.at_level(logging.INFO, logger="velari_ai.integrations.langchain.utils"):
        log_response(info, history=True)

    assert "earlier question" in caplog.text
    assert "earlier reply" in caplog.text
    assert "new reply" in caplog.text


def test_log_response_structured_output_logs_repr_without_pretty_print(caplog):
    import logging
    from pydantic import BaseModel
    from velari_ai.integrations.langchain.types import MessageStats, Metrics, ResponseInfo, UsageStats
    from velari_ai.integrations.langchain.utils import log_response

    class BalanceSummary(BaseModel):
        account_id: str
        balance: float

    response = BalanceSummary(account_id="ACC-1", balance=10.0)
    metrics = Metrics(
        latency_sec=0.5,
        usage_stats=UsageStats(input_tokens=10, output_tokens=5, reasoning_tokens=0),
        message_stats=MessageStats(cnt_total_messages=1, cnt_turn_messages=1, cnt_assistant=0),
    )
    info = ResponseInfo(response=response, metrics=metrics, messages=[])

    with caplog.at_level(logging.INFO, logger="velari_ai.integrations.langchain.utils"):
        log_response(info)

    assert "BalanceSummary" in caplog.text
    assert "ACC-1" in caplog.text
