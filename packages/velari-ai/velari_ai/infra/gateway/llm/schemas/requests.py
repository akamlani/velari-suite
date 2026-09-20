from    typing    import Any, Dict, List, Optional, Type
from    pydantic  import BaseModel, Field
# package modules
from    .types    import GatewayMessage


class GatewayRequest(BaseModel):
    """Provider-agnostic chat request — the one shape every ProviderAdapter accepts.

    Setting `response_model` requests structured output (see `GatewayResponse.response.parsed`)
    — adapters then ignore `tools`, and `stream()` doesn't support it.

    Examples:
        >>> request = GatewayRequest(
        ...     messages=[
        ...         GatewayMessage(role=Role.SYSTEM, content="You are a billing support assistant."),
        ...         GatewayMessage(role=Role.USER, content="What's the balance on ACC-10293?"),
        ...     ],
        ...     parameters={"temperature": 0.2},
        ... )
    """
    messages:       List[GatewayMessage]           = Field(description="Conversation turns, in order.")
    parameters:     Dict[str, Any]                 = Field(default_factory=dict, description="Provider-forwarded kwargs, e.g. temperature, max_tokens.")
    stream:         bool                           = Field(default=False, description="Whether the caller intends to consume this via stream() instead of chat().")
    tools:          Optional[List[Dict[str, Any]]] = Field(default=None, description="JSON-schema tool specs the model may call.")
    response_model: Optional[Type[BaseModel]]      = Field(default=None, description="Pydantic model for structured output; see the class docstring.")


class GatewayEmbeddingRequest(BaseModel):
    """Provider-agnostic embedding request — the one shape every EmbeddingAdapter accepts.

    Examples:
        >>> request = GatewayEmbeddingRequest(texts=["retrieval-augmented generation", "vector search"])
    """
    texts:      List[str]      = Field(description="Texts to embed.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Provider-forwarded kwargs.")
