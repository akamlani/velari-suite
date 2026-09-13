from dataclasses import dataclass, field
from pydantic import BaseModel, Field, create_model
from typing import Optional, List, Literal, Self, Sequence, TypedDict

######## Common Structures
class ResponseCitation(BaseModel):
    id: str    = Field(description="Identifier")
    title: str = Field(description="Title of the content identifier")

class ResponseReasoning(BaseModel):
    reasoning:  str = Field(..., description="Brief explanation of the WHY")
    confidence: str = Field(..., description="The level of confidence or conviction in the task's outcome.")

class ResponseMetrics(BaseModel):
    num_retries: int = Field(..., description="The number of retries attempted for the task.")
    num_turns:   int = Field(..., description="The number of turns in the conversation as proxy for time to completion.")

class ResponseGeneration(BaseModel):
    reasoning: ResponseReasoning      = Field(..., description="Brief explanation of how the answer was derived")
    metrics:   ResponseMetrics        = Field(..., description="Optional metrics related to the response generation.")
    citations: List[ResponseCitation] = Field(description="List of content identifiers cited in the answer")

class ResponseAnswer(BaseModel):
    reasoning: ResponseReasoning      = Field(..., description="Brief explanation of how the answer was derived")
    answer:    str                    = Field(..., description="The response answer to the user's question")
    metrics:   ResponseMetrics        = Field(..., description="Optional metrics related to the response generation.")
    citations: List[ResponseCitation] = Field(description="List of content identifiers cited in the answer")

######## Query Specification
class ResponseQueryRevision(BaseModel):
    """A revised search query, or a decision that no revision is possible."""
    reasoning: ResponseReasoning = Field(
        ..., description="The reasoning behind whether and how the query was revised."
    )
    can_revise: bool = Field(
        description="True if the question can be meaningfully revised into a better search query; "
        "False if the question is fundamentally unanswerable from this corpus."
    )
    query: Optional[str] = Field(
        default=None,
        description="The revised search query. Required when can_revise is True; omit otherwise.",
    )

class ResponseQueryDecomposition(BaseModel):
    """A complex query broken into atomic sub-queries, or a decision that decomposition isn't needed."""
    reasoning: ResponseReasoning = Field(
        ..., description="The reasoning behind whether the query should be decomposed into sub-queries."
    )
    should_decompose: bool = Field(
        description="True if the query should be split into multiple sub-queries; False if it's already a "
        "simple, single-lookup, or precise/technical query that would be harmed by splitting."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="Atomic sub-queries to retrieve independently. Empty when should_decompose is False.",
    )

class ResponseQueryExpansion(BaseModel):
    """Alternate phrasings of a query to broaden retrieval recall, or a decision that expansion isn't needed."""
    reasoning: ResponseReasoning = Field(
        ..., description="The reasoning behind whether the query should be expanded into alternate phrasings."
    )
    should_expand: bool = Field(
        description="True if alternate phrasings of the query would meaningfully broaden retrieval "
        "recall; False if the query is already precise/broad enough that paraphrasing wouldn't help."
    )
    expanded_queries: List[str] = Field(
        default_factory=list,
        description="Alternate phrasings preserving the same intent/scope. Empty when should_expand is False.",
    )

class ResponseQueryAugmentation(BaseModel):
    """A hypothetical answer passage synthesized to improve retrieval (HyDE), or a decision that augmentation isn't needed."""
    reasoning: ResponseReasoning = Field(
        ..., description="The reasoning behind whether a hypothetical answer passage would improve retrieval."
    )
    should_augment: bool = Field(
        description="True if a hypothetical answer passage would improve retrieval by better matching "
        "the vocabulary/structure of relevant documents; False for precise identifier-based lookups "
        "where fabricating specifics could mislead retrieval."
    )
    hypothetical_answer: Optional[str] = Field(
        default=None,
        description="A plausible, self-contained answer passage to embed for retrieval. None when should_augment is False.",
    )

######## Retrieval Grading
class ResponseRelevanceGrade(BaseModel):
    """Whether a single retrieved candidate is relevant enough to help answer the query."""
    reasoning: ResponseReasoning = Field(
        ..., description="The reasoning behind whether the retrieved context is relevant to the query."
    )
    is_relevant: bool = Field(
        description="True if the context contains information that would help answer the query; "
        "False if it is off-topic or would not contribute to an answer."
    )

######## Task Specification
class ResponseClassification(BaseModel):
    reasoning: ResponseReasoning = Field(..., description="The reasoning behind the classification.")
    task: str = Field(..., description="The name of the desired task to be performed.")
    description: str = Field(..., description="A brief description relative to the task objective.")
    category: str = Field(..., description="The predicted category label.")
    confidence: str = Field(..., description="The level of confidence or conviction in the predicted category.")

    # for sentiment analysis, let categories be values of Literal["positive", "negative", "neutral"]
    @classmethod
    def bind(cls, categories: List[str]) -> type[Self]:
        """Bind `category` to a closed set of allowed values, without editing the class definition.

        Args:
            categories (List[str]): Allowed category labels; `category` is validated against exactly these.

        Returns:
            type[Self]: A subclass with `category: Literal[*categories]`; `reasoning`/`confidence` are
                inherited unchanged.

        Examples:
            >>> BoundClassification = ResponseClassification.bind(
            ...     ["billing_question", "technical_issue", "account_access"]
            ... )
            >>> structured_model = model.with_structured_output(BoundClassification)
        """
        return create_model(
            f"{cls.__name__}Bound",
            __base__=cls,
            category=(Literal[tuple(categories)], ...),
        )

class ResponseTopic(BaseModel):
    class Topic(BaseModel):
        topic:       str = Field(..., description="The name of the predicted topic.")
        description: str = Field(..., description="A brief description of the predicted topic.")
        relevance: float = Field(
            ..., description="The relevance of the predicted topic to the overall content or context."
        )

    reasoning: ResponseReasoning = Field(..., description="The reasoning behind the topic classification.")
    topics: List[Topic]  = Field(..., description="The list of predicted topics.")

    def unpack(self) -> List[str]:
        """Unpack the list of topics into a list of unique topics."""
        return list({t.topic for t in self.topics})

class ResponseExtraction(BaseModel):
    class Entity(BaseModel):
        name: str = Field(..., description="The extracted entity's name or surface form.")
        type: str = Field(
            ...,
            description=(
                "The entity's category — standard NER types ('person', 'organization', 'location', "
                "'date', 'product') or a narrow domain/industry-specific NER type relevant to the text's "
                "subject matter, not generic terms or general techniques/methods."
            ),
        )

    class Concept(BaseModel):
        name:        str = Field(..., description="The concept's name.")
        description: str = Field(..., description="A brief description of the concept.")

    class Highlight(BaseModel):
        name:        str = Field(..., description="The highlighted phrase or sentence.")
        description: str = Field(..., description="Why this phrase or sentence is notable.")

    reasoning:  ResponseReasoning         = Field(description="The reasoning behind the extraction.")
    dates:      Optional[List[str]]       = Field(description="List of dates or time expressions found in the text.")
    keywords:   Optional[List[str]]       = Field(description="List of keywords extracted from the input text.")
    entities:   Optional[List[Entity]]    = Field(description="List of named entities extracted from the input text.")
    concepts:   Optional[List[Concept]]   = Field(description="List of concepts extracted from the input text.")
    highlights: Optional[List[Highlight]] = Field(description="List of highlighted phrases or sentences from the text.")

    @staticmethod
    def unpack(items: Optional[Sequence[Entity | Concept | Highlight]]) -> List[str]:
        """Unpack a structured list field into a list of unique names.

        Args:
            items (Optional[Sequence[Entity | Concept | Highlight]]): The field to unwrap — pass
                `self.entities`, `self.concepts`, or `self.highlights` directly.

        Returns:
            List[str]: Unique `.name` values from the given list, or `[]` if `None`/empty.

        Examples:
            >>> result = structured_model.invoke(prompt)
            >>> result.unpack(result.entities)
            >>> result.unpack(result.concepts)
            >>> result.unpack(result.highlights)
        """
        return list({item.name for item in items}) if items else []

class ResponseSynthesis(BaseModel):
    reasoning:    ResponseReasoning   = Field(..., description="The reasoning behind the synthesis.")
    summary:      str                 = Field(..., description="A concise summary of the main points.")
    preview:      Optional[str]       = Field(description="A brief preview or abstract snippet from the content.")
    key_insights: Optional[List[str]] = Field(description="List of key insights or findings from the content.")

class ResponseExtraction5Ws(BaseModel):
    who:   Optional[str] = Field(description="The person or entity involved in the event.")
    what:  Optional[str] = Field(description="The action or event that took place.")
    when:  Optional[str] = Field(description="The time or date when the event occurred.")
    where: Optional[str] = Field(description="The location where the event took place.")
    why:   Optional[str] = Field(description="The reason why the event occurred.")

class ResponseExtractionCrossRef(BaseModel):
    discrepancies:  Optional[List[str]] = Field(description="List of discrepancies found in the input text.")
    conflicts:      Optional[List[str]] = Field(description="List of conflicts found in the input text.")
    references:     Optional[List[str]] = Field(description="List of references found in the input text.")


######## Search Data models and Configuration
class ResponseSearchMetrics(BaseModel):
    num_searches: int = Field(..., description="The number of search queries executed.")
    num_fetches:  int = Field(..., description="The number of search fetch operations executed.")
    num_results:  int = Field(..., description="The total number of search results returned.")

class ResponseSearchResult(BaseModel):
    """One web search result returned by Tavily."""
    title:   str    = Field(description="The title of the search result.")
    url:     str    = Field(description="The URL of the search result.")
    content: str    = Field(description="The snippet or content of the search result.")
    preview: str    = Field(description="A short preview or abstract of the search result content.")
    score:   float  = Field(description="The relevance score of the search result, as determined by Tavily.")

class ResponseSearch(BaseModel):
    """Full response from a search — the answer summary plus individual results."""
    rationale: Optional[str]              = Field(description="The reasoning behind the answer summary.")
    query:     str                        = Field(description="The original search query.")
    results:   List[ResponseSearchResult] = Field(description="The list of search results returned by Tavily.")
    answer:    Optional[str]              = Field(description="The answer summary, if any.")
