from pydantic import BaseModel, Field

class EvalInfo(BaseModel):
    """Information about an evaluation run."""
    num_traces: int           = Field(..., description="The number of traces evaluated.")
    num_traces_feedback:  int = Field(..., description="The number of traces that received feedback.")
    num_traces_annotated: int = Field(..., description="The number of traces that received any annotation.")
    num_traces_comments:  int = Field(..., description="The number of traces that received comments.")
    num_traces_positive:  int = Field(..., description="The number of traces that received positive feedback.")
    num_traces_negative:  int = Field(..., description="The number of traces that received negative feedback.")

class EvalGrade(BaseModel):
    reasoning: str = Field(...,  description="Brief explanation of the WHY")
    relevant:  bool = Field(..., description="Whether the response is relevant to the question")
