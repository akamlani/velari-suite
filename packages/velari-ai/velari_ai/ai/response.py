from dataclasses import dataclass, field
from pydantic import BaseModel, Field, create_model
from typing import Optional, List, Literal, TypedDict

def _build__classification_model(name: str, categories: List[str]) -> type[BaseModel]:
    return create_model("Classification", category=(Literal[tuple(categories)], ...))

######## Common Structures
class Citation(BaseModel):
    id: str    = Field(description="Identifier")
    title: str = Field(description="Title of the content identifier")

class Reasoning(BaseModel):
    reasoning: str    = Field(..., description="Brief explanation of the WHY")
    confidence: float = Field(..., description="Confidence score of the task")

class ResponseGeneration(BaseModel):
    reasoning:  Reasoning      = Field(description="Brief explanation of how the answer was derived")
    answer:     str            = Field(description="The response answer to the user's question")
    citations:  List[Citation] = Field(description="List of content identifiers cited in the answer")

class Reflection(BaseModel):
    "Agent's reflection on the task execution after reading latest information, notes."
    knowledge_gap: Optional[str]    = Field(description="Identified areas of what is still unclear.")
    query_refinement: Optional[str] = Field(description="Refined query for follow-up search or retrieval.")


######## Risk Categorization Specification
# Risk Register: TBD
class RiskCategory(BaseModel):
    "Risk Scoring Detection is the task of classifying content into predefined risk categories based on the severity and potential impact."
    "Agent's assessment of risks, concerns, or potential issues with the task execution."
    category:    str = Field(..., description="Category of the identified risk or concern.")
    description: str = Field(..., description="Description of the risk category")
    severity:    Literal["P0", "P1", "P2", "P3", "P4"] = Field(..., description="Risk category label")

    def get_mapping(self) -> dict[str, str]:
        """Return a mapping of risk categories to their descriptions."""
        return {
            "P0": "Critical risk with immediate impact.",
            "P1": "High risk with significant impact.",
            "P2": "Moderate risk with noticeable impact.",
            "P3": "Low risk with minor impact.",
            "P4": "Minimal risk with negligible impact."
        }


######## Retrieval Specification
class RouterResponse(BaseModel):
    """The output of a router node, which determines the next node to execute based on the current state."""
    reasoning: str  = Field(..., description="Reasoning behind the routing decision.")
    collection: str = Field(..., description="Name of the table or collection or table to route query based on intent.")


######## Task Specification
class ResponseClassification(BaseModel):
    reasoning: Reasoning = Field(..., description="The reasoning behind the classification.")
    category:  str       = Field(..., description="The predicted category label.")








class ClassificationResult(BaseModel):
    category:   str
    confidence: str
    reasoning:  str

class ResponseTopics(BaseModel):
    class Topic(BaseModel):
        topic: str       = Field(..., description="The name of the predicted topic.")
        relevance: float = Field(..., description="The confidence score of the predicted topic.")

    reasoning: Reasoning = Field(..., description="The reasoning behind the topic classification.")
    topics: List[Topic]  = Field(..., description="The list of predicted topics.")

    def unpack(self) -> List[str]:
        """Unpack the list of topics into a list of unique topics."""
        return list({t.topic for t in self.topics})



class ExtractResult5Ws(BaseModel):
    who:    Optional[str] = Field(description="The person or entity involved in the event.")
    what:   Optional[str] = Field(description="The action or event that took place.")
    when:   Optional[str] = Field(description="The time or date when the event occurred.")
    where:  Optional[str] = Field(description="The location where the event took place.")
    why:    Optional[str] = Field(description="The reason why the event occurred.")


class ExtractResultCrossRef(BaseModel):
    discrepancies:  Optional[List[str]] = Field(description="List of discrepancies found in the input text.")
    conflicts:      Optional[List[str]] = Field(description="List of conflicts found in the input text.")
    references:     Optional[List[str]] = Field(description="List of references found in the input text.")

######## Evaluation Data models
class EvalGrade(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of the WHY")
    relevant: bool = Field(..., description="Whether the response is relevant to the question")




######## Search Data models and Configuration
@dataclass(frozen=True)
class SearchSpec:
    max_results_k: int = field(default=10)  # maximum search results to return for each search query

class SearchResult(BaseModel):
    """One web search result returned by Tavily."""
    title:   str    = Field(description="The title of the search result.")
    url:     str    = Field(description="The URL of the search result.")
    content: str    = Field(description="The snippet or content of the search result.")
    score:   float  = Field(description="The relevance score of the search result, as determined by Tavily.")

class SearchResponse(BaseModel):
    """Full response from a search — the answer summary plus individual results."""
    rationale: Optional[str]    = Field(description="The reasoning behind the answer summary.")
    query:   str                = Field(description="The original search query.")
    results: List[SearchResult] = Field(description="The list of search results returned by Tavily.")
    answer:  Optional[str]      = Field(description="The answer summary, if any.")
