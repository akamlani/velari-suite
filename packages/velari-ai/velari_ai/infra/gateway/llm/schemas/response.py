from    typing    import Any, List, Optional
from    pydantic  import BaseModel, Field
# package modules
from    .types import FinishReason, GatewayMessage
from    .....ai.types import ProviderName
from    .....ai.schemas.response import PerfMetrics, UsageMetrics


class GatewayResponse(BaseModel):
    """Provider-agnostic chat response — the one shape every ProviderAdapter returns.

    Fields are grouped into logical scopes — `model` (who served it), `metrics` (cost/timing),
    `response` (the actual reply) — rather than one flat list, so each scope can grow
    independently (e.g. `metrics` picking up cost/retry-count later) without reshuffling
    unrelated fields.

    Examples:
        >>> result = gateway.chat(request, model_config=ModelConfig(model="openai:gpt-4o-mini"))
        >>> result.response.message.content
        'Account ACC-10293 has an outstanding balance of $1,204.50.'
        >>> result.metrics.usage.total_tokens
        142

        >>> class AccountBalance(BaseModel):
        ...     account_id: str
        ...     balance_usd: float
        >>> structured_request = GatewayRequest(messages=request.messages, response_model=AccountBalance)
        >>> structured_result = gateway.chat(structured_request, model_config=ModelConfig(model="openai:gpt-4o-mini"))
        >>> structured_result.response.parsed
        AccountBalance(account_id='ACC-10293', balance_usd=1204.5)
    """
    class Model(BaseModel):
        provider: ProviderName = Field(description="Provider that served this response.")
        model:    str          = Field(description="Model identifier actually used.")

    class Metrics(BaseModel):
        perf:  PerfMetrics  = Field(description="Latency/performance accounting for this call.")
        usage: UsageMetrics = Field(description="Token accounting for this call.")

    class Response(BaseModel):
        message:       GatewayMessage      = Field(description="The model's reply.")
        finish_reason: FinishReason        = Field(description="Why the model stopped generating.")
        raw:           Optional[Any]       = Field(default=None, description="Escape hatch to the original provider SDK response object.")
        parsed:        Optional[BaseModel] = Field(default=None, description="Structured output instance, when request.response_model was set.")

    model:    Model    = Field(description="Provider/model identity for this response.")
    metrics:  Metrics  = Field(description="Latency and token-usage accounting for this call.")
    response: Response = Field(description="The model's reply, finish reason, and raw SDK payload.")


class GatewayEmbeddingResponse(BaseModel):
    """Provider-agnostic embedding response — the one shape every EmbeddingAdapter returns.

    Unlike `GatewayResponse` (a chat reply — message, finish reason, raw SDK payload), an
    embedding call has no conversational reply and no meaningful raw-payload escape hatch —
    just a vector per input text. So `result` holds only `embeddings`, not a `Response`-shaped
    bundle.

    Examples:
        >>> result = gateway.embed(["retrieval-augmented generation"], model_config=ModelConfig(model="openai:text-embedding-3-small"))
        >>> result.result.embedding[:3]
        [0.0123, -0.0456, 0.0789]
        >>> result.metrics.usage.total_tokens
        6
    """
    class Model(BaseModel):
        provider: ProviderName = Field(description="Provider that served this response.")
        model:    str          = Field(description="Model identifier actually used.")

    class Metrics(BaseModel):
        perf:  PerfMetrics  = Field(description="Latency/performance accounting for this call.")
        usage: UsageMetrics = Field(description="Token accounting for this call; zero for local models.")

    class Result(BaseModel):
        embeddings: List[List[float]] = Field(description="One embedding vector per input text, in order.")

        @property
        def embedding(self) -> List[float]:
            """The sole embedding vector, when exactly one text was embedded.

            Raises:
                RuntimeError: If more than one text was embedded — use `.embeddings` instead.
            """
            if len(self.embeddings) != 1:
                raise RuntimeError(
                    f"'.embedding' requires exactly one embedded text, got {len(self.embeddings)} — use '.embeddings' instead"
                )
            return self.embeddings[0]

    model:   Model   = Field(description="Provider/model identity for this response.")
    metrics: Metrics = Field(description="Latency and token-usage accounting for this call.")
    result:  Result  = Field(description="The embedding vectors.")
