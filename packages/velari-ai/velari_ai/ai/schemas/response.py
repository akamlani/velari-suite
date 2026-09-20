from    typing    import Optional
from    pydantic  import BaseModel, Field


class UsageMetrics(BaseModel):
    """Token accounting for one provider call — shared by chat and embedding calls alike.
    `output_tokens` is unset for embeddings (no generated output); `input_tokens` is unset
    when a provider only reports `total_tokens` (e.g. some local/embedding models).
    """
    input_tokens:  Optional[int] = Field(default=None, description="Tokens consumed by the request, when reported separately.")
    output_tokens: Optional[int] = Field(default=None, description="Tokens generated in the response; unset for embedding calls.")
    total_tokens:  int           = Field(description="Total tokens billed for this call.")


class PerfMetrics(BaseModel):
    """Call-level performance accounting — currently just latency, with room to grow (e.g.
    time-to-first-token for streaming, retry counts) without reshuffling callers.
    """
    latency_sec: float = Field(description="Wall-clock time spent in the adapter's/client's SDK call.")
