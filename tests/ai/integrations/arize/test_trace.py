"""Tests for velari_ai.integrations.arize.trace."""

import json

import pytest
from opentelemetry.trace import StatusCode


@pytest.fixture
def exporter():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


class TestTracingPipeline:
    def test_chain_span_sets_kind_and_input_value(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer, span_name="billing-pipeline")

        with pipeline.chain_span("summarize account ACC-10293"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "billing-pipeline"
        assert span.attributes["openinference.span.kind"] == "CHAIN"
        assert span.attributes["input.value"] == "summarize account ACC-10293"
        assert span.status.status_code == StatusCode.OK

    def test_chain_span_name_override(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer, span_name="billing-pipeline")

        with pipeline.chain_span("summarize account", span_name="custom-step"):
            pass

        assert exporter.get_finished_spans()[0].name == "custom-step"

    def test_chain_span_passes_through_extra_attrs(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer)

        with pipeline.chain_span("summarize account", account_id="ACC-10293"):
            pass

        assert exporter.get_finished_spans()[0].attributes["account_id"] == "ACC-10293"

    def test_chain_span_records_exception_and_sets_error_status(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer)

        with pytest.raises(ValueError):
            with pipeline.chain_span("summarize account"):
                raise ValueError("billing service unavailable")

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert [e.name for e in span.events] == ["exception"]
        assert span.events[0].attributes["exception.type"] == "ValueError"

    def test_llm_span_sets_kind_and_model_name(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer)

        with pipeline.llm_span("classify support ticket #4471", model="gpt-4o"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "llm"
        assert span.attributes["openinference.span.kind"] == "LLM"
        assert span.attributes["llm.model_name"] == "gpt-4o"
        assert span.status.status_code == StatusCode.OK

    def test_llm_span_records_exception_and_sets_error_status(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer)

        with pytest.raises(RuntimeError):
            with pipeline.llm_span("classify ticket", model="gpt-4o"):
                raise RuntimeError("model timeout")

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert [e.name for e in span.events] == ["exception"]

    def test_trace_llm_response_sets_all_attributes_when_given(self, tracer, exporter):
        from velari_ai.ai.tracing.trace import LLMCallResult
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer)
        result = LLMCallResult(
            output="order_status", prompt_tokens=37, completion_tokens=2,
            user_id="akamlani", metadata={"finish_reason": "stop"},
        )

        with pipeline.llm_span("Where is my order?", model="gpt-4o-mini") as span:
            pipeline.trace_llm_response(span, result)

        finished = exporter.get_finished_spans()[0]
        assert finished.attributes["output.value"] == "order_status"
        assert finished.attributes["llm.output_messages"] == "order_status"
        assert finished.attributes["llm.token_count.prompt"] == 37
        assert finished.attributes["llm.token_count.completion"] == 2
        assert finished.attributes["llm.token_count.total"] == 39
        assert finished.attributes["user.id"] == "akamlani"
        assert json.loads(finished.attributes["metadata"]) == {"finish_reason": "stop"}

    def test_trace_llm_response_omits_optional_attributes_when_not_given(self, tracer, exporter):
        from velari_ai.ai.tracing.trace import LLMCallResult
        from velari_ai.integrations.arize.trace import TracingPipeline

        pipeline = TracingPipeline(tracer)

        with pipeline.llm_span("Where is my order?", model="gpt-4o-mini") as span:
            pipeline.trace_llm_response(span, LLMCallResult(output="order_status"))

        finished = exporter.get_finished_spans()[0]
        assert finished.attributes["output.value"] == "order_status"
        assert "llm.token_count.prompt" not in finished.attributes
        assert "llm.token_count.completion" not in finished.attributes
        assert "llm.token_count.total" not in finished.attributes
        assert "user.id" not in finished.attributes
        assert "metadata" not in finished.attributes


class TestTracingRetrievalPipeline:
    def test_retriever_span_uses_configured_name_and_kind(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingRetrievalPipeline

        traced = TracingRetrievalPipeline(tracer, retriever_span_name="support-retrieval")

        with traced.retriever_span("What's our refund policy?"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "support-retrieval"
        assert span.attributes["openinference.span.kind"] == "RETRIEVER"

    def test_reranker_span_uses_configured_name_and_kind(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingRetrievalPipeline

        traced = TracingRetrievalPipeline(tracer, reranker_span_name="support-reranking")

        with traced.reranker_span("What's our refund policy?"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "support-reranking"
        assert span.attributes["openinference.span.kind"] == "RERANKER"

    def test_trace_span_sets_per_document_attributes(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingRetrievalPipeline

        traced = TracingRetrievalPipeline(tracer)
        results = [
            {"text": "We accept Visa and Mastercard.", "expected_category": "billing", "_score": 0.92},
            {"text": "Refunds are processed within 5 business days.", "expected_category": "billing", "_score": 0.81},
        ]

        with traced.retriever_span("What payment methods are accepted?") as span:
            traced.trace_span(span, results)

        finished = exporter.get_finished_spans()[0]
        assert finished.attributes["retrieval.documents.0.document.id"] == "0"
        assert finished.attributes["retrieval.documents.0.document.content"] == "We accept Visa and Mastercard."
        metadata = json.loads(finished.attributes["retrieval.documents.0.document.metadata"])
        assert metadata == {"expected_category": "billing", "_score": 0.92}
        assert finished.attributes["retrieval.documents.1.document.id"] == "1"

    def test_trace_span_uses_configured_text_field(self, tracer, exporter):
        from velari_ai.integrations.arize.trace import TracingRetrievalPipeline

        traced = TracingRetrievalPipeline(tracer)
        results = [{"content": "Visit /account/reset to reset your password.", "expected_category": "support"}]

        with traced.retriever_span("How do I reset my password?") as span:
            traced.trace_span(span, results, text_field="content")

        finished = exporter.get_finished_spans()[0]
        assert finished.attributes["retrieval.documents.0.document.content"] == "Visit /account/reset to reset your password."
        assert "content" not in json.loads(finished.attributes["retrieval.documents.0.document.metadata"])
