"""Tests for velari_ai.integrations.langchain.chunking."""

import pytest


def _long_document(source="https://example.com/doc"):
    from langchain_core.documents import Document

    return Document(page_content="word " * 400, metadata={"source": source})


def test_default_strategy_is_recursive_character():
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker()

    assert chunker._strategy == ChunkingStrategy.RECURSIVE_CHARACTER


def test_recursive_character_splits_into_multiple_chunks_preserving_metadata():
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=100, chunk_overlap=10)

    chunks = chunker.split_documents([_long_document()])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "https://example.com/doc" for chunk in chunks)


def test_character_strategy_splits_into_multiple_chunks():
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.CHARACTER, chunk_size=100, chunk_overlap=10, separator=" ")

    chunks = chunker.split_documents([_long_document()])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "https://example.com/doc" for chunk in chunks)


def test_token_strategy_splits_into_multiple_chunks():
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.TOKEN, chunk_size=20, chunk_overlap=5)

    chunks = chunker.split_documents([_long_document()])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "https://example.com/doc" for chunk in chunks)


def test_markdown_header_splits_on_headers_and_merges_original_metadata():
    from langchain_core.documents import Document
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.MARKDOWN_HEADER)
    markdown = "# Title\n\nIntro text.\n\n## Section A\n\nContent A here.\n\n## Section B\n\nContent B here."
    document = Document(page_content=markdown, metadata={"source": "readme.md"})

    chunks = chunker.split_documents([document])

    assert len(chunks) == 3
    assert all(chunk.metadata["source"] == "readme.md" for chunk in chunks)
    assert chunks[1].metadata["h1"] == "Title"
    assert chunks[1].metadata["h2"] == "Section A"
    assert chunks[1].page_content == "Content A here."


def test_markdown_header_accepts_custom_headers_to_split_on():
    from langchain_core.documents import Document
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.MARKDOWN_HEADER, headers_to_split_on=[("#", "title")])
    document = Document(page_content="# Only Header\n\nBody text.", metadata={"source": "notes.md"})

    chunks = chunker.split_documents([document])

    assert len(chunks) == 1
    assert chunks[0].metadata == {"source": "notes.md", "title": "Only Header"}


def test_split_text_returns_chunk_strings_for_recursive_character():
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=100, chunk_overlap=10)

    chunks = chunker.split_text("word " * 400)

    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_text_on_markdown_header_raises_typeerror():
    from velari_ai.integrations.langchain.chunking import ChunkingStrategy, DocumentChunker

    chunker = DocumentChunker(strategy=ChunkingStrategy.MARKDOWN_HEADER)

    with pytest.raises(TypeError):
        chunker.split_text("# Title\n\nBody text.")
