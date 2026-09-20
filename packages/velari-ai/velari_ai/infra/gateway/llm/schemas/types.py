from    datetime import datetime, timezone
from    enum     import StrEnum, auto
from    typing   import Any, Dict, List, Optional
from    pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM      = auto()
    USER        = auto()
    ASSISTANT   = auto()
    TOOL        = auto()


class FinishReason(StrEnum):
    STOP        = auto()
    TOOL_CALLS  = auto()
    LENGTH      = auto()
    ERROR       = auto()


class GatewayToolCall(BaseModel):
    id:        str            = Field(description="Provider-assigned identifier for this tool call.")
    name:      str            = Field(description="Name of the tool being called.")
    arguments: Dict[str, Any] = Field(description="Arguments the model supplied for the call.")


class GatewayMessage(BaseModel):
    role:         Role                            = Field(description="Who authored this message.")
    content:      str                             = Field(default="", description="Text content of the message.")
    tool_call_id: Optional[str]                   = Field(default=None, description="Set on a TOOL message — the GatewayToolCall.id it answers.")
    tool_calls:   Optional[List[GatewayToolCall]] = Field(default=None, description="Set on an ASSISTANT message that requests tool calls.")
    created_at:   datetime                        = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp this message was created.")
