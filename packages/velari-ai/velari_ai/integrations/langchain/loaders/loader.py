from __future__ import annotations

import  httpx
import  pandas as pd
from    datetime import datetime, timezone
from    pathlib import Path
from    typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, List
from    urllib.parse import urljoin

from    bs4 import BeautifulSoup

from    langchain_core.document_loaders import BaseLoader
from    langchain_core.documents import Document

# package modules
from    velari_core.core.io.filesystem import Filesystem
from    velari_core.core.io.partition.pdf import PartitionPdf
from    velari_core.core.io.types import ArtifactFormat
from    velari_core.core.services.client import HttpClient


class DocumentLoader(BaseLoader):
    @staticmethod
    def validate(docs: List[Document]) -> bool:
        try:
            return bool(docs) and all(bool(d.page_content) for d in docs)
        except Exception:
            return False

    @staticmethod
    def to_string(documents: List[Document]) -> str:
        """Format documents into a specific output format for building context for the prompt

        Args:
            documents: List of Document objects to format.

        Returns:
            Formatted output (e.g. string for prompt context).
        """
        return "\n\n".join([doc.page_content for doc in documents])

    @staticmethod
    def to_frame(documents: List[Document]) -> pd.DataFrame:
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
        self._uris = [web_paths] if isinstance(web_paths, str) else list(web_paths)
        self._bs_kwargs = {}
        self.client = HttpClient()

    def _scrape(self, url: str) -> Tuple[BeautifulSoup, str]:
        try:
            response = self.client.get(url)
        except httpx.HTTPError as e:
            raise RuntimeError(f"WebBaseLoader failed to load {url}: {e}") from e
        return BeautifulSoup(response.text, "html.parser", **self._bs_kwargs), str(response.url)

    # Add Section        (e.g., Title, Description, Chapters)
    # Add Source Type    (e.g., notes, report, experiment)
    # Add Type           (e.g., text, code, table, figure, image, video)
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
        if "description" in metadata:
            metadata["description_length"] = len(metadata["description"])

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

        metadata["content_length"] = len(page_content)
        metadata["acquired_at"]    = datetime.now(timezone.utc).isoformat()

        return metadata

    def lazy_load(self) -> Iterator[Document]:
        """Fetch and parse each URL in `web_paths`, yielding one `Document` per page.

        Returns:
            Iterator[Document]: `page_content` is the page's visible text; `metadata` fields
                are per `_enrich()`.

        Raises:
            RuntimeError: If fetching a URL fails (connection error, timeout, or HTTP error status).
        """
        for url in self._uris:
            soup, final_url = self._scrape(url)
            page_content = soup.get_text(separator=" ", strip=True)
            metadata = self._enrich(soup, final_url, page_content)
            yield Document(page_content=page_content, metadata=metadata)


class LocalFileLoader(DocumentLoader):
    """Load local files into Documents — dispatched by ArtifactFormat: PDF via PartitionPdf,
    CSV/Excel/JSON via Filesystem.parse(), YAML/text via Filesystem.read().

    Args:
        paths (Union[str, Sequence[str]]): One local file path, or several to load in sequence.
        text_field (Optional[str]): For CSV/Excel/JSON only — column to use as each row's page_content,
            yielding one Document per row (every other column becomes metadata, prefixed
            fields.<column> to avoid colliding with source/content_type/etc.). Omitted (default)
            keeps the whole table as one Document per file, serialized via `df.to_csv()`.

    Examples:
        >>> loader = LocalFileLoader(["README.md", "reports/quarterly_summary.pdf", "data/customers.csv"])
        >>> docs = loader.load()
        >>> docs[0].metadata["content_type"]
        'txt'
        >>> docs[2].metadata["num_rows"]
        150

        >>> faq_loader = LocalFileLoader("data/cs/datasets/faq.csv", text_field="question")
        >>> faq_docs = faq_loader.load()
        >>> faq_docs[0].page_content
        'How do I reset my password?'
        >>> faq_docs[0].metadata["fields.category"]
        'Account'
    """
    def __init__(self, paths: Union[str, Sequence[str]], text_field: Optional[str] = None) -> None:
        self._paths = [paths] if isinstance(paths, str) else list(paths)
        self._text_field = text_field

    def _make_document(self, path: str, fmt: Optional[ArtifactFormat], page_content: Any, **extra_metadata: Any) -> Document:
        content_type = fmt.value if fmt is not None else "text"
        metadata = {
            "source":         path,
            "content_type":   content_type,
            "content_length": len(page_content),
            "acquired_at":    datetime.now(timezone.utc).isoformat(),
            **extra_metadata,
        }
        return Document(page_content=page_content, metadata=metadata)

    def _tabular_documents(self, path: str, fmt: ArtifactFormat) -> List[Document]:
        df = Filesystem.parse(path)
        if self._text_field is None:
            return [self._make_document(
                path, fmt, df.to_csv(index=False), num_rows=len(df), num_columns=len(df.columns)
            )]

        if self._text_field not in df.columns:
            raise ValueError(f"text_field {self._text_field!r} not found in {path!r} columns: {list(df.columns)}")

        metadata_columns = [c for c in df.columns if c != self._text_field]
        return [
            self._make_document(
                path, fmt, str(text),
                **{f"fields.{col}": val for col, val in zip(metadata_columns, meta_values)},
            )
            for text, *meta_values in zip(df[self._text_field], *(df[c] for c in metadata_columns))
        ]

    def lazy_load(self) -> Iterator[Document]:
        """Read each path in `paths`, yielding one `Document` per file (or per row, with text_field).

        Returns:
            Iterator[Document]: `page_content` is the file's text (all pages joined for PDFs,
                CSV-serialized for CSV/Excel/JSON, `str`-serialized for YAML, or one row's
                text_field value per Document when text_field is set); `metadata` carries
                `source`/`content_type`/`content_length`/`acquired_at`, plus `num_rows`/`num_columns`
                for whole-table CSV/Excel/JSON, or every other column (prefixed `fields.<column>`)
                for text_field-based rows.
        """
        for path in self._paths:
            fmt = ArtifactFormat.from_ext(Path(path).suffix.lower())
            match fmt:
                case ArtifactFormat.PDF:
                    documents = [self._make_document(path, fmt, PartitionPdf(path).read())]
                case ArtifactFormat.CSV | ArtifactFormat.EXCEL | ArtifactFormat.JSON:
                    documents = self._tabular_documents(path, fmt)
                case ArtifactFormat.YAML:
                    documents = [self._make_document(path, fmt, str(Filesystem.read(path)))]
                case _:
                    documents = [self._make_document(path, fmt, Filesystem.read(path))]
            yield from documents
