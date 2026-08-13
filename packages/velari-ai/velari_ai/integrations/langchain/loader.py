from __future__ import annotations

import  httpx
import  pandas as pd
from    datetime import datetime, timezone
from    typing import Any, Dict, Iterator, Sequence, Tuple, Union, List
from    urllib.parse import urljoin

from    bs4 import BeautifulSoup

from    langchain_core.document_loaders import BaseLoader
from    langchain_core.documents import Document

# package modules
from    velari_core.core.services.client import HttpClient


class DocumentLoader(BaseLoader):
    def validate(self, docs: List[Document]) -> bool:
        try:
            return bool(docs) and all(bool(d.page_content) for d in docs)
        except Exception:
            return False

    def to_string(self, documents: List[Document]) -> str:
        """Format documents into a specific output format for building context for the prompt

        Args:
            documents: List of Document objects to format.

        Returns:
            Formatted output (e.g. string for prompt context).
        """
        return "\n\n".join([doc.page_content for doc in documents])

    def to_frame(self, documents: List[Document]) -> pd.DataFrame:
        """Flatten documents into a single DataFrame — one row per document.

        Args:
            documents (List[Document]): Documents to flatten; each metadata key becomes a
                `meta.<key>` column.

        Returns:
            pd.DataFrame: `page_content` plus one `meta.<key>` column per metadata key found
                across `documents`.

        Examples:
            >>> loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/document_loaders")
            >>> df_docs = loader.to_frame(loader.load())
            >>> df_docs[["meta.source", "meta.title", "meta.length"]]
        """
        df_documents = pd.DataFrame(
            {"page_content": doc.page_content, **{f"meta.{k}": v for k, v in doc.metadata.items()}}
            for doc in documents
        )
        return df_documents


class WebBaseLoader(DocumentLoader):
    """Load and parse web pages into `Document`s — one per URL, via `HttpClient` + BeautifulSoup.

    Synchronous and sequential — no retries, rate-limiting, or concurrent fetching.

    Args:
        web_paths (Union[str, Sequence[str]]): One URL, or several to load in sequence.

    Examples:
        >>> loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/document_loaders")
        >>> docs = loader.load()
        >>> docs[0].metadata["title"]
        'Document loader integrations - Docs by LangChain'
    """
    def __init__(self, web_paths: Union[str, Sequence[str]]) -> None:
        self.web_paths = [web_paths] if isinstance(web_paths, str) else list(web_paths)
        self.client = HttpClient()

    def _scrape(self, url: str) -> Tuple[BeautifulSoup, str]:
        try:
            response = self.client.get(url)
        except httpx.HTTPError as e:
            raise RuntimeError(f"WebBaseLoader failed to load {url}: {e}") from e
        return BeautifulSoup(response.text, "html.parser"), str(response.url)

    # Add Section (Title, Description, Chapters)
    # Add Source Type (e.g., notes, report, experiment)
    # Add Type (e.g., text, code, table, figure, image, video)
    # Add File Extension (e.g., .txt, .md, .pdf, .csv, .json, .png, .jpg, .mp4)
    # Add Collection Key to Document Metadata
    @staticmethod
    def _enrich(soup: BeautifulSoup, url: str, page_content: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"source": url}

        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        elif og_title := soup.find("meta", attrs={"property": "og:title"}):
            metadata["title"] = str(og_title.get("content", ""))

        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = str(description.get("content", ""))
        elif og_description := soup.find("meta", attrs={"property": "og:description"}):
            metadata["description"] = str(og_description.get("content", ""))

        if section := soup.find("meta", attrs={"property": "article:section"}):
            metadata["topic"] = str(section.get("content", ""))
        elif keywords := soup.find("meta", attrs={"name": "keywords"}):
            first_keyword = str(keywords.get("content", "")).split(",")[0].strip()
            if first_keyword:
                metadata["topic"] = first_keyword

        if html := soup.find("html"):
            metadata["language"] = str(html.get("lang", ""))

        links = list(dict.fromkeys(
            urljoin(url, str(a.get("href"))) for a in soup.find_all("a", href=True)
        ))
        if links:
            metadata["links"] = links

        metadata["length"]      = len(page_content)
        metadata["acquired_at"] = datetime.now(timezone.utc).isoformat()

        return metadata

    def lazy_load(self) -> Iterator[Document]:
        """Fetch and parse each URL in `web_paths`, yielding one `Document` per page.

        Returns:
            Iterator[Document]: `page_content` is the page's visible text; `metadata` fields
                are per `_enrich()`.

        Raises:
            RuntimeError: If fetching a URL fails (connection error, timeout, or HTTP error status).
        """
        for url in self.web_paths:
            soup, final_url = self._scrape(url)
            page_content = soup.get_text(separator=" ", strip=True)
            metadata = self._enrich(soup, final_url, page_content)
            yield Document(page_content=page_content, metadata=metadata)
