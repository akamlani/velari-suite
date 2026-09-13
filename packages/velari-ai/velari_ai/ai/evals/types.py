from pydantic import BaseModel, Field

class EvalTraceInfo(BaseModel):
    """Information about an evaluation run."""
    num_traces: int           = Field(..., description="The number of traces evaluated.")
    num_traces_feedback:  int = Field(..., description="The number of traces that received feedback.")
    num_traces_annotated: int = Field(..., description="The number of traces that received any annotation.")
    num_traces_comments:  int = Field(..., description="The number of traces that received comments.")
    num_traces_positive:  int = Field(..., description="The number of traces that received positive feedback.")
    num_traces_negative:  int = Field(..., description="The number of traces that received negative feedback.")

class EvalMetrics(BaseModel):
    num_turns: int      = Field(..., description="The number of turns evaluated.")
    num_tool_calls: int = Field(..., description="The number of tool calls evaluated.")
