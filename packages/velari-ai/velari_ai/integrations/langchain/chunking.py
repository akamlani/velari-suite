from __future__ import annotations

from    enum import StrEnum, auto
from    typing import Any, List, Union

from langchain_text_splitters import (
    # Splits text into chunks based on a specified number of characters.
    # Useful for consistent chunk sizes regardless of content structure.
    CharacterTextSplitter,
    # Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
    # Ideal for maintaining semantic coherence within chunks.
    SentenceTransformersTokenTextSplitter,
    # Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
    # Useful for transformer models with strict token limits.
    TokenTextSplitter,
    # Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
    # Balances between maintaining coherence and adhering to character limits.
    RecursiveCharacterTextSplitter,
    # Splits text based on Markdown headers, preserving the structure of the document.
    # Useful for documents with clear sectioning, like technical documentation or reports.
    MarkdownHeaderTextSplitter,
    # Splits text into chunks based on a specified number of words.
    # Useful for word-based processing or when word count is more relevant than character count.
    TextSplitter,
)
from langchain_core.documents import Document


class ChunkingStrategy(StrEnum):
    CHARACTER                   = auto()
    TOKEN                       = auto()
    SENTENCE_TRANSFORMER_TOKEN  = auto()
    RECURSIVE_CHARACTER         = auto()
    MARKDOWN_HEADER             = auto()


class DocumentChunker(object):
    """Split text or Documents into chunks via a configurable langchain_text_splitters strategy.

    Use `split_text()` for raw strings, `split_documents()` for `Document`s (preserves/merges
    metadata) — `MARKDOWN_HEADER` only supports `split_documents()`, since its output metadata is
    the header hierarchy alone and needs merging with the original document's own metadata.

    `SENTENCE_TRANSFORMER_TOKEN` requires the optional `sentence-transformers` package.

    Args:
        strategy (ChunkingStrategy): Which splitter to use; defaults to `RECURSIVE_CHARACTER`
            (attempts natural boundaries — sentences/paragraphs — within `chunk_size`).
        chunk_size (int): Target chunk size in characters or tokens, per strategy. Not forwarded
            for `MARKDOWN_HEADER` (no fixed size) or `SENTENCE_TRANSFORMER_TOKEN` (use
            `tokens_per_chunk` via `**kwargs` instead — its real size knob).
        chunk_overlap (int): Overlap between adjacent chunks, to preserve context across boundaries.
        **kwargs (Any): Strategy-specific constructor kwargs, forwarded to the underlying splitter:
            - `CHARACTER`: `separator` (default `"\n\n"`), `is_separator_regex`.
            - `TOKEN`: `encoding_name` (default `"gpt2"`), `model_name`, `allowed_special`,
              `disallowed_special`.
            - `SENTENCE_TRANSFORMER_TOKEN`: `model_name`, `tokens_per_chunk`, `model_kwargs`.
            - `RECURSIVE_CHARACTER`: `separators`, `keep_separator`, `is_separator_regex`.
            - `MARKDOWN_HEADER`: `headers_to_split_on` (default `h1`/`h2`/`h3`),
              `return_each_line`, `strip_headers`, `custom_header_patterns`.

    Examples:
        >>> loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/document_loaders")
        >>> chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.split_documents(loader.load())
    """
    _MARKDOWN_DEFAULT_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        chunk_size:    int = 512,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> None:
        self._strategy = strategy
        self._splitter: Union[TextSplitter, MarkdownHeaderTextSplitter] = self._build_splitter(
            strategy, chunk_size, chunk_overlap, **kwargs,
        )

    def _build_splitter(
        self,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
        **kwargs: Any,
    ) -> Union[TextSplitter, MarkdownHeaderTextSplitter]:
        match strategy:
            case ChunkingStrategy.CHARACTER:
                return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
            case ChunkingStrategy.TOKEN:
                return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
            case ChunkingStrategy.SENTENCE_TRANSFORMER_TOKEN:
                return SentenceTransformersTokenTextSplitter(chunk_overlap=chunk_overlap, **kwargs)
            case ChunkingStrategy.MARKDOWN_HEADER:
                headers_to_split_on = kwargs.pop("headers_to_split_on", self._MARKDOWN_DEFAULT_HEADERS)
                return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, **kwargs)
            case _:
                return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text: str) -> List[str]:
        """Split a raw text string into chunks per the configured strategy.

        Args:
            text (str): Raw text to split.

        Returns:
            List[str]: Chunk strings.

        Raises:
            TypeError: If the configured strategy is `MARKDOWN_HEADER` — its splitter returns
                `Document`s carrying header metadata, not plain strings; use `split_documents()`
                instead, which is the only method that can preserve that metadata.

        Examples:
            >>> chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=200, chunk_overlap=20)
            >>> chunks = chunker.split_text("Retrieval-augmented generation combines a retriever with a generator model.")
        """
        if self._strategy == ChunkingStrategy.MARKDOWN_HEADER:
            raise TypeError(
                "MARKDOWN_HEADER splits into Documents carrying header metadata, not plain strings — "
                "use split_documents() instead"
            )
        assert isinstance(self._splitter, TextSplitter)
        return self._splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split `documents` into chunks per the configured strategy.

        Returns:
            List[Document]: One or more chunks per input document. `MARKDOWN_HEADER` chunks carry
                the original document's metadata merged with the header hierarchy (`h1`/`h2`/...
                keys) `MarkdownHeaderTextSplitter` adds per chunk — it has no `split_documents()`
                of its own, so this merge is done manually to avoid losing the source metadata.

        Examples:
            >>> chunker = DocumentChunker(strategy=ChunkingStrategy.MARKDOWN_HEADER)
            >>> chunks = chunker.split_documents([Document(page_content="# Title\\n\\nIntro.", metadata={"source": "readme.md"})])
            >>> chunks[0].metadata
            {'source': 'readme.md', 'h1': 'Title'}
        """
        if self._strategy == ChunkingStrategy.MARKDOWN_HEADER:
            assert isinstance(self._splitter, MarkdownHeaderTextSplitter)
            return [
                Document(page_content=chunk.page_content, metadata={**doc.metadata, **chunk.metadata})
                for doc in documents
                for chunk in self._splitter.split_text(doc.page_content)
            ]
        assert isinstance(self._splitter, TextSplitter)
        return self._splitter.split_documents(documents)
