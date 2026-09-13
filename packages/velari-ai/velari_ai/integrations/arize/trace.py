from __future__ import annotations

import  json
import  logging
from    contextlib import contextmanager
from    typing import Any, ContextManager, Dict, Iterator, List, Optional

from    openinference.semconv.trace import OpenInferenceSpanKindValues
from    opentelemetry import trace
from    phoenix.otel import SpanAttributes
# package modules
from    ...ai.retrieval.vectorstore import SearchResult
from    ...ai.tracing.trace import LLMCallResult

logger = logging.getLogger(__name__)


class TracingPipeline(object):
    """Wrap an OpenTelemetry tracer with OpenInference-annotated span helpers.

    Args:
        tracer (trace.Tracer): OpenTelemetry tracer to start spans on.
        span_name (str): Default name for `chain_span()` when no override is given.

    Examples:
        >>> # Entering a span's `with` block inside another's makes it a child span —
        >>> # OpenTelemetry tracks the "current span" via context, not the call site.
        >>> pipeline = TracingPipeline(tracer, span_name="support-ticket-pipeline")
        >>> with pipeline.chain_span("Triage support ticket #4471") as chain:  # root span
        ...     with pipeline.llm_span(
        ...         "Classify support ticket #4471", model="gpt-4o", span_name="classify-intent",
        ...     ) as intent:  # child of chain
        ...         intent_response = llm_client.invoke("Classify support ticket #4471")
        ...     with pipeline.llm_span(
        ...         "Draft a reply for ticket #4471", model="gpt-4o", span_name="draft-reply",
        ...     ) as reply:  # also a child of chain
        ...         reply_response = llm_client.invoke("Draft a reply for ticket #4471")
        ... # resulting span tree:
        ... #   support-ticket-pipeline (CHAIN, root)
        ... #     +-- classify-intent (LLM)
        ... #     +-- draft-reply (LLM)
    """
    def __init__(self, tracer: trace.Tracer, span_name: str = "pipeline") -> None:
        self.tracer    = tracer
        self.span_name = span_name

    @contextmanager
    def _span(
        self,
        span_name: str,
        kind: OpenInferenceSpanKindValues,
        input_value: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[trace.Span]:
        with self.tracer.start_as_current_span(
            span_name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value,
                SpanAttributes.INPUT_VALUE: input_value,
                **(attributes or {}),
            },
        ) as span:
            yield span
            span.set_status(trace.Status(trace.StatusCode.OK))

    def chain_span(self, input_value: str, span_name: Optional[str] = None, **extra_attrs: Any) -> ContextManager[trace.Span]:
        """Start a CHAIN-kind span for a pipeline step.

        Args:
            input_value (str): Recorded as the span's `input.value` attribute.
            span_name (Optional[str]): Overrides `self.span_name` for this call.
            **extra_attrs (Any): Additional span attributes.

        Returns:
            ContextManager[trace.Span]: Active span, status set to OK on success (OTel's
                start_as_current_span() records the exception and sets ERROR automatically
                on failure).

        Examples:
            >>> pipeline = TracingPipeline(tracer, span_name="billing-pipeline")
            >>> with pipeline.chain_span("Summarize account ACC-10293's billing history") as span:
            ...     summary = summarize_billing_history("ACC-10293")
        """
        return self._span(span_name or self.span_name, OpenInferenceSpanKindValues.CHAIN, input_value, extra_attrs)

    def llm_span(self, input_value: str, model: str, span_name: str = "llm", **extra_attrs: Any) -> ContextManager[trace.Span]:
        """Start an LLM-kind span for a model call.

        Args:
            input_value (str): Recorded as the span's `input.value` attribute.
            model (str): Recorded as the span's `llm.model_name` attribute.
            span_name (str): Span name.
            **extra_attrs (Any): Additional span attributes.

        Returns:
            ContextManager[trace.Span]: Active span, status set to OK on success (OTel's
                start_as_current_span() records the exception and sets ERROR automatically
                on failure).

        Examples:
            >>> pipeline = TracingPipeline(tracer)
            >>> with pipeline.llm_span("Classify support ticket #4471", model="gpt-4o") as span:
            ...     response = llm.invoke("Classify support ticket #4471")
        """
        return self._span(span_name, OpenInferenceSpanKindValues.LLM, input_value, {SpanAttributes.LLM_MODEL_NAME: model, **extra_attrs})

    def trace_llm_response(self, span: trace.Span, result: LLMCallResult) -> None:
        """Annotate an LLM span with a call's output, token usage, user, and metadata.

        Provider/framework-agnostic — `result` is a plain `LLMCallResult`, not tied to any
        specific LLM client library. Token-count/user/metadata attributes are only set when
        the corresponding `result` field is given.

        Args:
            span (trace.Span): Active span to annotate — typically from `llm_span()`.
            result (LLMCallResult): Output text, and optionally prompt/completion token counts,
                user id, and a metadata dict.

        Examples:
            >>> pipeline = TracingPipeline(tracer)
            >>> with pipeline.llm_span("Classify support ticket #4471", model="gpt-4o") as span:
            ...     response = llm.invoke("Classify support ticket #4471")
            ...     pipeline.trace_llm_response(span, LLMCallResult(
            ...         output=response.content, prompt_tokens=42, completion_tokens=8,
            ...     ))
        """
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.output)
        span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, result.output)
        if result.prompt_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, result.prompt_tokens)
        if result.completion_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, result.completion_tokens)
        if result.prompt_tokens is not None and result.completion_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, result.prompt_tokens + result.completion_tokens)
        if result.user_id is not None:
            span.set_attribute(SpanAttributes.USER_ID, result.user_id)
        if result.metadata:
            span.set_attribute(SpanAttributes.METADATA, json.dumps(result.metadata))


class TracingRetrievalPipeline(TracingPipeline):
    """Extend `TracingPipeline` with RETRIEVER/RERANKER spans for a RAG pipeline.

    Args:
        tracer (trace.Tracer): OpenTelemetry tracer to start spans on.
        chain_span_name (str): Default name for the overall pipeline's `chain_span()`.
        retriever_span_name (str): Name for `retriever_span()`.
        reranker_span_name (str): Name for `reranker_span()`.

    Examples:
        >>> # Entering a span's `with` block inside another's makes it a child span —
        >>> # OpenTelemetry tracks the "current span" via context, not the call site,
        >>> # so nesting here is what creates the parent/child relationship below.
        >>> traced = TracingRetrievalPipeline(tracer, chain_span_name="rag-pipeline")
        >>> with traced.chain_span("What's our refund policy?") as chain:        # root span
        ...     with traced.retriever_span("What's our refund policy?") as r:    # child of chain
        ...         results = pipeline.retriever.search("What's our refund policy?", top_k=5)
        ...         traced.trace_span(r, results)
        ...     with traced.llm_span("Answer using retrieved context", model="gpt-4o") as llm:  # also a child of chain
        ...         response = llm_client.invoke("Answer using retrieved context")
        ... # resulting span tree:
        ... #   rag-pipeline (CHAIN, root)
        ... #     +-- rag-retrieval (RETRIEVER)
        ... #     +-- llm (LLM)
    """
    def __init__(
        self,
        tracer: trace.Tracer,
        chain_span_name:     str = "rag-pipeline",
        retriever_span_name: str = "rag-retrieval",
        reranker_span_name:  str = "rag-reranking",
    ) -> None:
        super().__init__(tracer, chain_span_name)
        self._retriever_span_name = retriever_span_name
        self._reranker_span_name  = reranker_span_name

    def retriever_span(self, input_value: str, **extra_attrs: Any) -> ContextManager[trace.Span]:
        """Start a RETRIEVER-kind span for a document-retrieval step.

        Args:
            input_value (str): Recorded as the span's `input.value` attribute — typically the query.
            **extra_attrs (Any): Additional span attributes.

        Returns:
            ContextManager[trace.Span]: Active span, status set to OK on success (OTel's
                start_as_current_span() records the exception and sets ERROR automatically
                on failure).

        Examples:
            >>> traced = TracingRetrievalPipeline(tracer)
            >>> with traced.retriever_span("What's our refund policy?") as span:
            ...     results = pipeline.retriever.search("What's our refund policy?", top_k=5)
            ...     traced.trace_span(span, results)
        """
        return self._span(self._retriever_span_name, OpenInferenceSpanKindValues.RETRIEVER, input_value, extra_attrs)

    def reranker_span(self, input_value: str, **extra_attrs: Any) -> ContextManager[trace.Span]:
        """Start a RERANKER-kind span for a reranking step.

        Args:
            input_value (str): Recorded as the span's `input.value` attribute — typically the query.
            **extra_attrs (Any): Additional span attributes.

        Returns:
            ContextManager[trace.Span]: Active span, status set to OK on success (OTel's
                start_as_current_span() records the exception and sets ERROR automatically
                on failure).

        Examples:
            >>> traced = TracingRetrievalPipeline(tracer)
            >>> with traced.reranker_span("What's our refund policy?") as span:
            ...     results = pipeline.retrieve_and_rerank("What's our refund policy?", reranker=reranker, top_k=3)
            ...     traced.trace_span(span, results)
        """
        return self._span(self._reranker_span_name, OpenInferenceSpanKindValues.RERANKER, input_value, extra_attrs)

    def trace_span(self, span: trace.Span, results: List[SearchResult], text_field: str = "text") -> None:
        """Annotate a span with OpenInference document attributes for each retrieved result.

        Sets three attributes per document: a positional ``document.id``, the document
        content from the configured text field, and a JSON-serialised metadata dict
        containing all remaining fields (scores, category, etc.).

        Args:
            span (trace.Span): Active OpenTelemetry span to annotate — typically a
                retriever or reranker span obtained from ``retriever_span`` /
                ``reranker_span``.
            results (List[SearchResult]): Retrieved or reranked documents. Each dict
                contains three categories of fields: (1) ``"text"`` — the document
                content; (2) source-document metadata fields passed through unchanged
                (e.g. ``expected_category``, any dataset column except the embedding);
                (3) score fields added by the retriever or reranker (``_score``,
                ``_distance``, ``_relevance``, ``_rerank_score``).

        Examples:
            >>> # Retrieval result shape: text + source metadata + _score
            >>> # results[0] == {"text": "We accept Visa and Mastercard...",
            >>> #                "expected_category": "billing", "_score": 0.92}
            >>> traced = TracingRetrievalPipeline(tracer)
            >>> with traced.retriever_span(query) as r_span:
            ...     results = pipeline.retriever.search(query, top_k=5)
            ...     traced.trace_span(r_span, results)
            ... # r_span attributes set for each doc (i=0 shown):
            ... #   retrieval.documents.0.document.id       -> "0"
            ... #   retrieval.documents.0.document.content  -> "We accept Visa and Mastercard..."
            ... #   retrieval.documents.0.document.metadata -> '{"expected_category": "billing", "_score": 0.92}'
            >>> # After reranking — _rerank_score added alongside _score in metadata
            >>> with traced.reranker_span(query) as rr_span:
            ...     results = pipeline.retrieve_and_rerank(query, reranker=reranker, top_k=3)
            ...     traced.trace_span(rr_span, results)
            ... # rr_span metadata: '{"expected_category": "billing", "_score": 0.78, "_rerank_score": 0.95}'
        """
        for i, r in enumerate(results):
            span.set_attribute(f"retrieval.documents.{i}.document.id", str(i))
            span.set_attribute(
                f"retrieval.documents.{i}.document.content",
                str(r.get(text_field, "")),
            )
            span.set_attribute(
                f"retrieval.documents.{i}.document.metadata",
                json.dumps({k: v for k, v in r.items() if k != text_field}),
            )
