"""Tests for velari_ai.integrations.langchain.types."""


def test_agentresponseinfo_holds_response_and_metrics():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.types import (
        AgentResponseInfo, ToolCallMessageStats, ToolCallMetrics, UsageStats,
    )

    metrics = ToolCallMetrics(
        latency_sec=0.5,
        message_stats=ToolCallMessageStats(
            cnt_turn_messages=1, cnt_total_messages=1,
            cnt_assistant=1, cnt_tool_results=0, cnt_tool_request_turns=0, cnt_tool_requests=0,
        ),
        usage_stats=UsageStats(input_tokens=10, output_tokens=5, reasoning_tokens=0),
    )
    info = AgentResponseInfo(response=AIMessage(content="ok"), metrics=metrics, messages=[AIMessage(content="ok")])

    assert info.text == "ok"
    assert info.metrics.latency_sec == 0.5
    assert info.metrics.message_stats.cnt_assistant == 1
    assert info.metrics.usage_stats.input_tokens == 10


def test_responseinfo_text_returns_content_for_aimessage_response():
    from langchain_core.messages import AIMessage
    from velari_ai.integrations.langchain.types import MessageStats, Metrics, ResponseInfo, UsageStats

    metrics = Metrics(
        latency_sec=0.5,
        usage_stats=UsageStats(input_tokens=10, output_tokens=5, reasoning_tokens=0),
        message_stats=MessageStats(cnt_total_messages=1, cnt_turn_messages=1, cnt_assistant=1),
    )
    response = AIMessage(content="Account ACC-1 balance: $10")
    info = ResponseInfo(response=response, metrics=metrics, messages=[response])

    assert info.text == "Account ACC-1 balance: $10"


def test_responseinfo_text_raises_for_structured_response():
    import pytest
    from pydantic import BaseModel
    from velari_ai.integrations.langchain.types import MessageStats, Metrics, ResponseInfo, UsageStats

    class BalanceSummary(BaseModel):
        account_id: str
        balance: float

    metrics = Metrics(
        latency_sec=0.5,
        usage_stats=UsageStats(input_tokens=10, output_tokens=5, reasoning_tokens=0),
        message_stats=MessageStats(cnt_total_messages=1, cnt_turn_messages=1, cnt_assistant=0),
    )
    response = BalanceSummary(account_id="ACC-1", balance=10.0)
    info = ResponseInfo(response=response, metrics=metrics, messages=[])

    with pytest.raises(RuntimeError):
        info.text


def test_messagestats_from_messages_counts_general_turn():
    from langchain_core.messages import AIMessage, HumanMessage
    from velari_ai.integrations.langchain.types import MessageStats

    messages = [HumanMessage(content="hi"), AIMessage(content="Your balance is $10.")]

    stats = MessageStats.from_messages(messages, cnt_total_messages=2)

    assert stats.cnt_assistant == 1
    assert stats.cnt_turn_messages == 2
    assert stats.cnt_total_messages == 2


def test_messagestats_from_messages_defaults_cnt_total_messages_to_turn_length():
    from langchain_core.messages import AIMessage, HumanMessage
    from velari_ai.integrations.langchain.types import MessageStats

    messages = [HumanMessage(content="hi"), AIMessage(content="Your balance is $10.")]

    stats = MessageStats.from_messages(messages)

    assert stats.cnt_total_messages == len(messages)


def test_toolcallmessagestats_from_messages_counts_tool_calling_turn():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from velari_ai.integrations.langchain.types import ToolCallMessageStats

    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{
            "name": "lookup_account_balance", "args": {"account_id": "ACC-1"},
            "id": "call_1", "type": "tool_call",
        }]),
        ToolMessage(content="Account ACC-1 balance: $10", tool_call_id="call_1"),
        AIMessage(content="Your balance is $10."),
    ]

    stats = ToolCallMessageStats.from_messages(messages, cnt_total_messages=4)

    assert stats.cnt_assistant == 2
    assert stats.cnt_tool_results == 1
    assert stats.cnt_tool_request_turns == 1
    assert stats.cnt_tool_requests == 1
    assert stats.cnt_total_messages == 4


def test_toolcallmessagestats_from_messages_defaults_cnt_total_messages_to_turn_length():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from velari_ai.integrations.langchain.types import ToolCallMessageStats

    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{
            "name": "lookup_account_balance", "args": {"account_id": "ACC-1"},
            "id": "call_1", "type": "tool_call",
        }]),
        ToolMessage(content="Account ACC-1 balance: $10", tool_call_id="call_1"),
        AIMessage(content="Your balance is $10."),
    ]

    stats = ToolCallMessageStats.from_messages(messages)

    assert stats.cnt_total_messages == len(messages)


def test_usagestats_from_messages_sums_usage_metadata():
    from typing import List
    from langchain_core.messages import AIMessage, BaseMessage
    from velari_ai.integrations.langchain.types import UsageStats

    messages: List[BaseMessage] = [AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
            "output_token_details": {"reasoning": 12},
        },
    )]

    stats = UsageStats.from_messages(messages)

    assert stats.input_tokens == 120
    assert stats.output_tokens == 30
    assert stats.reasoning_tokens == 12
