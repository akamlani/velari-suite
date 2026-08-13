"""Tests for velari_ai.integrations.langchain.loader."""

import pytest


def test_load_returns_document_with_text_and_metadata(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = (
            "<html lang='en'><head><title>Example</title>"
            "<meta name='description' content='An example page.'></head>"
            "<body><p>Hello world</p></body></html>"
        )

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == "https://example.com"
    assert docs[0].metadata["title"] == "Example"
    assert docs[0].metadata["description"] == "An example page."
    assert docs[0].metadata["language"] == "en"
    assert docs[0].metadata["length"] == len(docs[0].page_content)


def test_load_sets_acquired_at_to_current_utc_time(monkeypatch):
    from datetime import datetime, timezone
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = "<html><title>Example</title><body>ok</body></html>"

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    before = datetime.now(timezone.utc)
    docs = loader.load()
    after = datetime.now(timezone.utc)

    acquired_at = datetime.fromisoformat(docs[0].metadata["acquired_at"])
    assert before <= acquired_at <= after


def test_to_frame_flattens_documents_into_one_row_each(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        def __init__(self, url, text):
            self.url = url
            self.text = text

    responses = {
        "https://example.com/a": _FakeResponse(
            "https://example.com/a", "<html lang='en'><title>A</title><body>Page A</body></html>",
        ),
        "https://example.com/b": _FakeResponse(
            "https://example.com/b", "<html lang='en'><title>B</title><body>Page B</body></html>",
        ),
    }
    loader = WebBaseLoader(["https://example.com/a", "https://example.com/b"])
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: responses[url])
    docs = loader.load()

    df_docs = loader.to_frame(docs)

    assert len(df_docs) == 2
    assert list(df_docs["meta.title"]) == ["A", "B"]
    assert list(df_docs["page_content"]) == [d.page_content for d in docs]
    assert list(df_docs["meta.length"]) == [d.metadata["length"] for d in docs]


def test_load_multiple_urls_yields_one_document_each(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        def __init__(self, url, text):
            self.url = url
            self.text = text

    responses = {
        "https://example.com/a": _FakeResponse(
            "https://example.com/a", "<html><title>A</title><body>Page A</body></html>",
        ),
        "https://example.com/b": _FakeResponse(
            "https://example.com/b", "<html><title>B</title><body>Page B</body></html>",
        ),
    }
    loader = WebBaseLoader(["https://example.com/a", "https://example.com/b"])
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: responses[url])

    docs = loader.load()

    assert [d.metadata["title"] for d in docs] == ["A", "B"]


def test_load_propagates_failed_fetch_as_runtimeerror(monkeypatch):
    import httpx
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    def _raise(url, **kwargs):
        raise httpx.HTTPError("404")

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", _raise)

    with pytest.raises(RuntimeError):
        loader.load()


def test_load_uses_final_url_after_redirect(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com/redirected"
        text = "<html><title>Redirected</title><body>ok</body></html>"

    loader = WebBaseLoader("http://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert docs[0].metadata["source"] == "https://example.com/redirected"


def test_load_falls_back_to_open_graph_tags(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = (
            "<html><head>"
            "<meta property='og:title' content='OG Title'>"
            "<meta property='og:description' content='OG description.'>"
            "</head><body>ok</body></html>"
        )

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert docs[0].metadata["title"] == "OG Title"
    assert docs[0].metadata["description"] == "OG description."


def test_load_extracts_topic_from_article_section(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = (
            "<html><head>"
            "<meta property='article:section' content='Cybersecurity'>"
            "<meta name='keywords' content='xss, csrf, sql injection'>"
            "</head><body>ok</body></html>"
        )

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert docs[0].metadata["topic"] == "Cybersecurity"


def test_load_falls_back_to_first_keyword_for_topic(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = (
            "<html><head>"
            "<meta name='keywords' content='xss, csrf, sql injection'>"
            "</head><body>ok</body></html>"
        )

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert docs[0].metadata["topic"] == "xss"


def test_load_extracts_deduped_absolute_links(monkeypatch):
    from velari_ai.integrations.langchain.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com/docs/"
        text = (
            "<html><body>"
            "<a href='/about'>About</a>"
            "<a href='https://other.com/page'>Other</a>"
            "<a href='/about'>About again</a>"
            "</body></html>"
        )

    loader = WebBaseLoader("https://example.com/docs/")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert docs[0].metadata["links"] == ["https://example.com/about", "https://other.com/page"]
