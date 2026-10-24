from __future__ import annotations

import  httpx
import  pandas as pd
from    dataclasses import asdict
from    datetime import datetime, timezone
from    pathlib import Path
from    typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, List
from    urllib.parse import urljoin, urlparse

from    bs4 import BeautifulSoup

from    langchain_core.document_loaders import BaseLoader
from    langchain_core.documents import Document

# package modules
from    velari_core.core.io.filesystem import Filesystem
from    velari_core.core.io.partition.pdf import PartitionPdf
from    velari_core.core.io.types import ArtifactFormat
from    velari_core.core.services.client import HttpClient
from    . import metadata as md


class DocumentLoader(BaseLoader):
    PREVIEW_CHARS = 500

    def _describe(self, metadata: md.Metadata, page_content: str, *, uri: str, relative: str, is_remote: bool) -> None:
        """Fill in the base metadata every document has, on whichever `Metadata` subclass the loader builds.

        Args:
            metadata (md.Metadata): `PageMetadata`/`FileMetadata` instance to populate in place.
            page_content (str): Extracted text, used for the preview and `content.metrics.length`.
            uri (str): Where the content came from (final URL or absolute file path).
            relative (str): Short form of `uri`: the path as given for files, the URL path (+ `#section`) for pages.
            is_remote (bool): Whether `uri` was fetched over the network. Also decides how `source.info.name` is read:
                the file stem for a local path, the last URL path segment (else the host) for a page.

        Examples:
            >>> loader = LocalFileLoader("reports/quarterly_summary.pdf")
            >>> metadata = md.FileMetadata()
            >>> loader._describe(
            ...     metadata, "Q3 churn fell to 2.1%", uri="/srv/reports/quarterly_summary.pdf",
            ...     relative="reports/quarterly_summary.pdf", is_remote=False,
            ... )
            >>> metadata.source.properties.extension
            'pdf'
        """
        path = Path(urlparse(uri).path if is_remote else uri)
        metadata.source.info.uri = uri
        metadata.source.info.relative = relative
        metadata.source.info.name = path.stem or urlparse(uri).netloc
        metadata.source.properties.extension = path.suffix.lstrip(".").lower() or None
        metadata.source.properties.is_remote = is_remote
        metadata.content.preview = page_content[:self.PREVIEW_CHARS]
        metadata.content.metrics.length = len(page_content)
        metadata.timestamp.acquired_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _flatten(record: md.Metadata) -> Dict[str, Any]:
        """Unpack a metadata record into the flat dotted dict a `Document.metadata` needs.

        Args:
            record (md.Metadata): Populated `PageMetadata`/`FileMetadata`; unset (`None`) fields are skipped.

        Returns:
            Dict[str, Any]: e.g. `{"source.info.relative": ..., "page.title": ..., "record.fields.<column>": ...}`.

        Examples:
            >>> metadata = md.FileMetadata()
            >>> metadata.source.info.relative = "data/cs/datasets/faq.csv"
            >>> metadata.record.fields["category"] = "Account"
            >>> DocumentLoader._flatten(metadata)
            {'source.info.relative': 'data/cs/datasets/faq.csv', 'record.fields.category': 'Account'}
        """
        row = pd.json_normalize(asdict(record), sep=".").to_dict("records")[0]
        return {str(key): value for key, value in row.items() if value is not None}

    def enrich(self, documents: List[Document]) -> List[Document]:
        """Add further metadata to already loaded documents, in place (step 2 after `load()`'s core metadata).

        Args:
            documents (List[Document]): Documents this loader loaded; their `metadata` is updated, no new `Document` is created.

        Returns:
            List[Document]: The same list, so calls chain. The base loader has nothing further to add.

        Examples:
            >>> loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/document_loaders")
            >>> docs = loader.enrich(loader.load())
            >>> docs[0].metadata["page.title"]
            'Document loader integrations - Docs by LangChain'
        """
        return documents

    @staticmethod
    def validate(docs: List[Document]) -> bool:
        """Whether `docs` is non-empty and every document has text or is a table row (has `record.index`)."""
        try:
            return bool(docs) and all(d.page_content or "record.index" in d.metadata for d in docs)
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
            >>> df_docs = loader.to_frame(loader.enrich(loader.load()))
            >>> df_docs[["meta.source.info.uri", "meta.page.title", "meta.content.metrics.length"]]
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
        >>> docs[0].metadata["source.info.uri"]
        'https://docs.langchain.com/oss/python/integrations/document_loaders'
        >>> docs = loader.enrich(docs)
        >>> docs[0].metadata["page.title"]
        'Document loader integrations - Docs by LangChain'
    """
    def __init__(self, web_paths: Union[str, Sequence[str]]) -> None:
        self._uris = [web_paths] if isinstance(web_paths, str) else list(web_paths)
        self._bs_kwargs = {}
        self._soups: Dict[str, BeautifulSoup] = {}
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
    def _page(self, soup: BeautifulSoup, uri: str) -> md.PageMetadata:
        metadata = md.PageMetadata()
        page = metadata.page

        if title := soup.find("title"):
            page.title = title.get_text()
        elif og_title := soup.find("meta", attrs={"property": "og:title"}):
            page.title = str(og_title.get("content", ""))

        if description := soup.find("meta", attrs={"name": "description"}):
            page.description = str(description.get("content", ""))
        elif og_description := soup.find("meta", attrs={"property": "og:description"}):
            page.description = str(og_description.get("content", ""))
        if page.description is not None:
            page.description_length = len(page.description)

        if section := soup.find("meta", attrs={"property": "article:section"}):
            page.topic = str(section.get("content", ""))
        elif keywords := soup.find("meta", attrs={"name": "keywords"}):
            first_keyword = str(keywords.get("content", "")).split(",")[0].strip()
            if first_keyword:
                page.topic = first_keyword

        if html := soup.find("html"):
            page.language = str(html.get("lang", ""))

        links = list(dict.fromkeys(
            urljoin(uri, str(a.get("href"))) for a in soup.find_all("a", href=True)
        ))
        if links:
            page.links = links

        return metadata

    def enrich(self, documents: List[Document]) -> List[Document]:
        """Add the page's `page.*` metadata (title, description, topic, language, links) to documents this loader loaded.

        Args:
            documents (List[Document]): Documents from this loader's `load()`; matched to their page by `source.info.uri`
                and updated in place. Documents it did not load are left untouched; repeating the call is harmless.

        Returns:
            List[Document]: The same list, so calls chain.

        Examples:
            >>> loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
            >>> docs = loader.enrich(loader.load())
            >>> docs[0].metadata["page.title"]
            "LLM Powered Autonomous Agents | Lil'Log"
        """
        for doc in documents:
            uri = doc.metadata.get("source.info.uri")
            if uri in self._soups:
                doc.metadata.update(self._flatten(self._page(self._soups[uri], uri)))
        return documents

    def lazy_load(self) -> Iterator[Document]:
        """Fetch and parse each URL in `web_paths`, yielding one `Document` per page with core metadata only.

        Returns:
            Iterator[Document]: `page_content` is the page's visible text; `metadata` is the flattened core
                `PageMetadata` (`source.*`, `content.*`, `timestamp.*`); `page.*` is added later by `enrich()`.

        Raises:
            RuntimeError: If fetching a URL fails (connection error, timeout, or HTTP error status).
        """
        for url in self._uris:
            soup, final_url = self._scrape(url)
            page_content = soup.get_text(separator=" ", strip=True)
            parts = urlparse(final_url)
            relative = parts.path.strip("/") + (f"#{parts.fragment}" if parts.fragment else "")
            metadata = md.PageMetadata()
            self._describe(metadata, page_content, uri=final_url, relative=relative, is_remote=True)
            self._soups[final_url] = soup
            yield Document(page_content=page_content, metadata=self._flatten(metadata))


class LocalFileLoader(DocumentLoader):
    """Load local files into Documents — dispatched by ArtifactFormat: PDF via PartitionPdf,
    CSV/Excel/JSON via Filesystem.parse(), YAML/text via Filesystem.read().

    `load()` gives core metadata only; for CSV/Excel/JSON, call `enrich()` to add the file's shape
    (`source.table.*`) and each row's columns (`record.fields.*`).

    Args:
        paths (Union[str, Sequence[str]]): One local file path, or several to load in sequence.
        text_field (Optional[str]): For CSV/Excel/JSON only — column to use as each row's page_content.
            Every row is one Document either way, numbered by `record.index`; `enrich()` adds every other column
            as record-level metadata, prefixed record.fields.<column> to avoid colliding with source.*/content.*/etc.
            (blank cells are omitted). Omitted (default): there is no text, so `page_content` is empty and
            all columns end up in `record.fields`.
        base_dir (Optional[str]): Directory that relative paths resolve against; defaults to the working directory.
            Each document records the path as given in `source.info.relative` and its absolute form in `source.info.uri`.

    Examples:
        >>> loader = LocalFileLoader(["README.md", "reports/quarterly_summary.pdf", "data/customers.csv"])
        >>> docs = loader.enrich(loader.load())
        >>> docs[0].metadata["source.properties.extension"]
        'md'
        >>> docs[2].metadata["source.table.num_rows"]
        150

        >>> loader = LocalFileLoader("data/cs/datasets/orders.csv")
        >>> orders = loader.enrich(loader.load())
        >>> orders[0].page_content, orders[0].metadata["record.fields.order_id"]
        ('', 'ORD-12345')

        >>> faq_loader = LocalFileLoader("data/cs/datasets/faq.csv", text_field="question")
        >>> faq_docs = faq_loader.enrich(faq_loader.load())
        >>> faq_docs[0].page_content
        'How do I reset my password?'
        >>> faq_docs[0].metadata["record.fields.category"]
        'Account'
        >>> faq_docs[0].metadata["source.info.relative"]
        'data/cs/datasets/faq.csv'
    """
    def __init__(
        self, paths: Union[str, Sequence[str]], text_field: Optional[str] = None, base_dir: Optional[str] = None
    ) -> None:
        self._paths      = [paths] if isinstance(paths, str) else list(paths)
        self._text_field = text_field
        self._base_dir   = base_dir
        self._tables: Dict[str, pd.DataFrame] = {}

    def _metadata(self, page_content: str, *, path: str, relpath: str, index: Optional[int] = None) -> md.FileMetadata:
        metadata = md.FileMetadata()
        self._describe(metadata, page_content, uri=str(Path(path).absolute()), relative=relpath, is_remote=False)
        metadata.record.index = index
        return metadata

    def _make_document(self, path: str, relpath: str, page_content: Any, index: Optional[int] = None) -> Document:
        metadata = self._metadata(page_content, path=path, relpath=relpath, index=index)
        return Document(page_content=page_content, metadata=self._flatten(metadata))

    def _tabular_documents(self, path: str, relpath: str) -> List[Document]:
        df = Filesystem.parse(path)
        if self._text_field is not None and self._text_field not in df.columns:
            raise ValueError(f"text_field {self._text_field!r} not found in {path!r} columns: {list(df.columns)}")
        self._tables[str(Path(path).absolute())] = df
        texts = [str(text) for text in df[self._text_field]] if self._text_field is not None else [""] * len(df)
        return [self._make_document(path, relpath, text, index=index) for index, text in enumerate(texts)]

    def enrich(self, documents: List[Document]) -> List[Document]:
        """Add the file's shape (`source.table.*`) and each row's columns (`record.fields.*`) to the documents this loader loaded.

        Args:
            documents (List[Document]): Table-row documents from this loader's `load()`; matched to their row by
                `source.info.uri` and `record.index`, and updated in place. Other documents are left untouched;
                repeating the call is harmless.

        Returns:
            List[Document]: The same list, so calls chain.

        Examples:
            >>> loader = LocalFileLoader("data/cs/datasets/orders.csv")
            >>> orders = loader.enrich(loader.load())
            >>> orders[0].metadata["source.table.num_rows"], orders[0].metadata["record.fields.status"]
            (20, 'shipped')
        """
        for doc in documents:
            uri, index = doc.metadata.get("source.info.uri"), doc.metadata.get("record.index")
            if uri in self._tables and index is not None:
                df = self._tables[uri]
                row = df.iloc[[index]].to_dict("records")[0]
                row.pop(self._text_field, None)
                metadata = md.FileMetadata()
                metadata.source.table = md.FileMetadata.Source.Table(num_rows=len(df), num_columns=len(df.columns))
                metadata.record.fields = {str(key): value for key, value in row.items() if not (pd.api.types.is_scalar(value) and pd.isna(value))}
                doc.metadata.update(self._flatten(metadata))
        return documents

    def lazy_load(self) -> Iterator[Document]:
        """Read each path in `paths`, yielding one `Document` per file (or per row, for CSV/Excel/JSON) with core metadata only.

        Returns:
            Iterator[Document]: `page_content` is the file's text (all pages joined for PDFs,
                `str`-serialized for YAML, or for CSV/Excel/JSON one row's text_field value, empty when
                there is no text_field); `metadata` is the flattened core `FileMetadata` (`source.*`, `content.*`,
                `timestamp.*`), plus `record.index` for table rows. `source.table.*` and `record.fields.*` are added
                later by `enrich()`.
        """
        for relpath in self._paths:
            path = str(Path(self._base_dir) / relpath) if self._base_dir else relpath
            fmt = ArtifactFormat.from_ext(Path(path).suffix.lower())
            match fmt:
                case ArtifactFormat.PDF:
                    documents = [self._make_document(path, relpath, PartitionPdf(path).read())]
                case ArtifactFormat.CSV | ArtifactFormat.EXCEL | ArtifactFormat.JSON:
                    documents = self._tabular_documents(path, relpath)
                case ArtifactFormat.YAML:
                    documents = [self._make_document(path, relpath, str(Filesystem.read(path)))]
                case _:
                    documents = [self._make_document(path, relpath, Filesystem.read(path))]
            yield from documents


class AutoLoader(DocumentLoader):
    """Load a mix of URLs and local files, picking `WebBaseLoader` or `LocalFileLoader` per source from its value.

    Args:
        sources (Sequence[str]): `http(s)://` URLs load as web pages; anything else is a local file path.
        text_field (Optional[str]): Forwarded to `LocalFileLoader` for CSV/Excel/JSON sources; ignored for URLs.
        base_dir (Optional[str]): Directory that relative local paths resolve against; defaults to the working directory.

    Examples:
        >>> loader = AutoLoader(["https://lilianweng.github.io/posts/2023-06-23-agent/", "reports/quarterly_summary.pdf"])
        >>> docs = loader.load()
        >>> docs[0].metadata["source.info.uri"]
        'https://lilianweng.github.io/posts/2023-06-23-agent/'
        >>> docs[1].metadata["source.info.relative"]
        'reports/quarterly_summary.pdf'
    """
    def __init__(self, sources: Sequence[str], text_field: Optional[str] = None, base_dir: Optional[str] = None) -> None:
        self._sources     = list(sources)
        self._text_field  = text_field
        self._base_dir    = base_dir
        self._loaders: List[DocumentLoader] = []

    def _loader_for(self, source: str) -> DocumentLoader:
        match urlparse(source).scheme:
            case "http" | "https":
                return WebBaseLoader(source)
            case _:
                return LocalFileLoader(source, text_field=self._text_field, base_dir=self._base_dir)

    def lazy_load(self) -> Iterator[Document]:
        """Load each source with its inferred loader, yielding documents in source order.

        Returns:
            Iterator[Document]: Whatever the chosen loader yields, with that loader's core metadata.

        Raises:
            RuntimeError: If fetching a URL fails (from `WebBaseLoader`).
        """
        self._loaders = []
        for source in self._sources:
            loader = self._loader_for(source)
            self._loaders.append(loader)
            yield from loader.lazy_load()

    def enrich(self, documents: List[Document]) -> List[Document]:
        """Add further metadata, in place, through each inner loader that loaded these documents.

        Args:
            documents (List[Document]): Documents from this loader's `load()`; each inner loader only updates its own.

        Returns:
            List[Document]: The same list, so calls chain.

        Examples:
            >>> loader = AutoLoader(["https://lilianweng.github.io/posts/2023-06-23-agent/", "data/cs/datasets/faq.csv"])
            >>> docs = loader.enrich(loader.load())
            >>> docs[0].metadata["page.title"]
            "LLM Powered Autonomous Agents | Lil'Log"
        """
        for loader in self._loaders:
            loader.enrich(documents)
        return documents
