from __future__ import annotations

from    typing import Literal

from    tavily import TavilyClient

# local files
from    ...ai.response import SearchResponse, SearchResult


class TavilySearch(object):
    """Web search — grounds answers in current, real-world information via the Tavily API.

    Args:
        api_key (str): Tavily API key (`TAVILY_API_KEY` in `.env`).

    Examples:
        >>> searcher = TavilySearch(api_key="tvly-...")
        >>> response = searcher.search("latest developments in retrieval-augmented generation")
        >>> response.results[0].url
        'https://example.com/rag-2026-overview'
    """
    def __init__(self, api_key: str) -> None:
        self._client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_answer: bool = False,
    ) -> SearchResponse:
        """Search the web for current, relevant information to ground answers in evidence.

        Search the internet for any query — facts, news, documentation, how-to guides, or research on any subject.
        Returns source results and, when available, a direct answer.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return; defaults to 5.
            search_depth (Literal["basic", "advanced", "fast", "ultra-fast"]): `"advanced"`
                trades latency for recall; `"fast"`/`"ultra-fast"` trade recall for latency.
            include_answer (bool): Also ask Tavily to generate a short summary answer.

        Returns:
            SearchResponse: `results` (title/url/content/score per hit) and, when
                `include_answer=True`, an `answer` summary string.

        Raises:
            RuntimeError: If the Tavily API call fails.

        Examples:
            >>> searcher = TavilySearch(api_key="tvly-...")
            >>> response = searcher.search("retrieval-augmented generation trends", max_results=3)
            >>> [r.title for r in response.results]
            ['RAG in 2026: A Practical Overview', 'Advances in Retrieval-Augmented Generation']
        """
        try:
            raw = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
            )
        except Exception as e:
            raise RuntimeError(f"search() failed to query Tavily: {e}") from e
        results = [
            SearchResult(title=r["title"], url=r["url"], content=r["content"], score=r["score"])
            for r in raw.get("results", [])
        ]
        return SearchResponse(
            query     = query,
            results   = results,
            rationale = raw.get("rationale"),
            answer    = raw.get("answer"),
        )
