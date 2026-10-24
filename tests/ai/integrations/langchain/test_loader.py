"""Tests for velari_ai.integrations.langchain.loaders.loader."""

import pytest


def test_load_returns_document_with_text_and_metadata(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = (
            "<html lang='en'><head><title>Example</title>"
            "<meta name='description' content='An example page.'></head>"
            "<body><p>Hello world</p></body></html>"
        )

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.enrich(loader.load())

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source.info.uri"] == "https://example.com"
    assert docs[0].metadata["page.title"] == "Example"
    assert docs[0].metadata["page.description"] == "An example page."
    assert docs[0].metadata["page.language"] == "en"
    assert docs[0].metadata["content.metrics.length"] == len(docs[0].page_content)


def test_load_sets_acquired_at_to_current_utc_time(monkeypatch):
    from datetime import datetime, timezone
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = "<html><title>Example</title><body>ok</body></html>"

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    before = datetime.now(timezone.utc)
    docs = loader.load()
    after = datetime.now(timezone.utc)

    acquired_at = datetime.fromisoformat(docs[0].metadata["timestamp.acquired_at"])
    assert before <= acquired_at <= after


def test_to_frame_flattens_documents_into_one_row_each(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

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
    docs = loader.enrich(loader.load())

    df_docs = loader.to_frame(docs)

    assert len(df_docs) == 2
    assert list(df_docs["meta.page.title"]) == ["A", "B"]
    assert list(df_docs["page_content"]) == [d.page_content for d in docs]
    assert list(df_docs["meta.content.metrics.length"]) == [d.metadata["content.metrics.length"] for d in docs]


def test_load_multiple_urls_yields_one_document_each(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

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

    docs = loader.enrich(loader.load())

    assert [d.metadata["page.title"] for d in docs] == ["A", "B"]


def test_load_propagates_failed_fetch_as_runtimeerror(monkeypatch):
    import httpx
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    def _raise(url, **kwargs):
        raise httpx.HTTPError("404")

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", _raise)

    with pytest.raises(RuntimeError):
        loader.load()


@pytest.mark.parametrize("url, relative", [
    ("https://example.com", ""),
    ("https://example.com/posts/agent/#planning", "posts/agent#planning"),
])
def test_load_records_url_path_and_section_as_source_relative(monkeypatch, url, relative):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    class _FakeResponse:
        def __init__(self, url):
            self.url = url
            self.text = "<html><title>Agents</title><body>Agent design notes.</body></html>"

    loader = WebBaseLoader(url)
    monkeypatch.setattr(loader.client, "get", lambda _url, **kwargs: _FakeResponse(url))

    metadata = loader.load()[0].metadata

    assert metadata["source.info.relative"] == relative
    assert metadata["source.properties.is_remote"] is True
    assert "source.properties.extension" not in metadata


def test_load_uses_final_url_after_redirect(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com/redirected"
        text = "<html><title>Redirected</title><body>ok</body></html>"

    loader = WebBaseLoader("http://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.load()

    assert docs[0].metadata["source.info.uri"] == "https://example.com/redirected"


def test_load_falls_back_to_open_graph_tags(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

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

    docs = loader.enrich(loader.load())

    assert docs[0].metadata["page.title"] == "OG Title"
    assert docs[0].metadata["page.description"] == "OG description."


def test_load_extracts_topic_from_article_section(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

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

    docs = loader.enrich(loader.load())

    assert docs[0].metadata["page.topic"] == "Cybersecurity"


def test_load_falls_back_to_first_keyword_for_topic(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    class _FakeResponse:
        url = "https://example.com"
        text = (
            "<html><head>"
            "<meta name='keywords' content='xss, csrf, sql injection'>"
            "</head><body>ok</body></html>"
        )

    loader = WebBaseLoader("https://example.com")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakeResponse())

    docs = loader.enrich(loader.load())

    assert docs[0].metadata["page.topic"] == "xss"


def test_load_extracts_deduped_absolute_links(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

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

    docs = loader.enrich(loader.load())

    assert docs[0].metadata["page.links"] == ["https://example.com/about", "https://other.com/page"]


def test_auto_loader_picks_loader_per_source_type(monkeypatch, tmp_path):
    from velari_core.core.services.client import HttpClient
    from velari_ai.integrations.langchain.loaders.loader import AutoLoader

    class _FakeResponse:
        url = "https://example.com/report"
        text = "<html><title>Report</title><body>Quarterly report</body></html>"

    monkeypatch.setattr(HttpClient, "get", lambda self, url, **kwargs: _FakeResponse())
    notes = tmp_path / "notes.txt"
    notes.write_text("Churn fell in Q3.")

    loader = AutoLoader(["https://example.com/report", str(notes)])
    docs = loader.enrich(loader.load())

    assert docs[0].metadata["source.info.uri"] == "https://example.com/report"
    assert docs[1].metadata["source.info.uri"] == str(notes)
    assert docs[0].metadata["page.title"] == "Report"
    assert docs[1].metadata["source.properties.extension"] == "txt"
    assert [d.metadata["source.properties.is_remote"] for d in docs] == [True, False]
    assert "page.title" not in docs[1].metadata


def test_auto_loader_resolves_relative_paths_against_base_dir_and_forwards_text_field(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import AutoLoader

    (tmp_path / "faq.csv").write_text("question,category\nHow do I reset my password?,Account\n")

    loader = AutoLoader(["faq.csv"], text_field="question", base_dir=str(tmp_path))
    docs = loader.enrich(loader.load())

    assert docs[0].page_content == "How do I reset my password?"
    assert docs[0].metadata["record.fields.category"] == "Account"


def test_auto_loader_records_source_info_relative_and_uri_for_files(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import AutoLoader

    (tmp_path / "faq.csv").write_text("question,category\nHow do I reset my password?,Account\n")

    metadata = AutoLoader(["faq.csv"], text_field="question", base_dir=str(tmp_path)).load()[0].metadata

    assert metadata["source.info.relative"] == "faq.csv"
    assert metadata["source.info.uri"] == str(tmp_path / "faq.csv")
    assert "source" not in metadata


def test_flatten_keeps_dotted_field_columns_and_list_values_and_drops_unset_keys():
    from velari_ai.integrations.langchain.loaders.loader import DocumentLoader
    from velari_ai.integrations.langchain.loaders.metadata import FileMetadata, PageMetadata

    file_record = FileMetadata()
    file_record.record.fields["order.id"] = "A-1"
    page_record = PageMetadata()
    page_record.page.links = ["https://example.com/about"]

    assert DocumentLoader._flatten(file_record) == {"record.fields.order.id": "A-1"}
    assert DocumentLoader._flatten(page_record) == {"page.links": ["https://example.com/about"]}


def test_row_documents_carry_both_file_shape_and_record_fields(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import LocalFileLoader

    path = tmp_path / "faq.csv"
    path.write_text("question,category\nHow do I reset my password?,Account\nWhat is your refund policy?,Billing\n")

    loader = LocalFileLoader(str(path), text_field="question")
    metadata = loader.enrich(loader.load())[1].metadata

    assert metadata["source.table.num_rows"] == 2
    assert metadata["source.table.num_columns"] == 2
    assert metadata["record.fields.category"] == "Billing"


def test_without_text_field_each_row_is_a_document_with_all_columns_as_record_fields(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import LocalFileLoader

    path = tmp_path / "orders.csv"
    path.write_text("order_id,status,quantity,unit_price,carrier\nORD-1,shipped,2,9.5,FedEx\nORD-2,pending,1,20.0,\n")

    loader = LocalFileLoader(str(path))
    docs = loader.enrich(loader.load())

    assert [d.page_content for d in docs] == ["", ""]
    assert docs[0].metadata["record.fields.order_id"] == "ORD-1"
    assert docs[0].metadata["record.fields.quantity"] == 2
    assert docs[0].metadata["record.fields.unit_price"] == 9.5
    assert docs[1].metadata["source.table.num_rows"] == 2
    assert "record.fields.carrier" not in docs[1].metadata


def test_file_metadata_source_extends_base_source_in_one_attribute():
    from velari_ai.integrations.langchain.loaders.loader import DocumentLoader
    from velari_ai.integrations.langchain.loaders.metadata import FileMetadata, Metadata

    metadata = FileMetadata()
    metadata.source.info.relative = "data/cs/datasets/faq.csv"
    metadata.source.properties.extension = "csv"
    metadata.source.table.num_rows = 20

    assert issubclass(FileMetadata.Source, Metadata.Source)
    assert DocumentLoader._flatten(metadata) == {
        "source.info.relative": "data/cs/datasets/faq.csv",
        "source.properties.extension": "csv",
        "source.table.num_rows": 20,
    }


class _FakePage:
    def __init__(self, url="https://example.com/agents"):
        self.url = url
        self.text = "<html lang='en'><title>Agents</title><body>Agent design notes.</body></html>"


def test_load_gives_core_metadata_only_until_enriched(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    loader = WebBaseLoader("https://example.com/agents")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakePage())

    metadata = loader.load()[0].metadata

    assert metadata["source.info.uri"] == "https://example.com/agents"
    assert "content.metrics.length" in metadata
    assert not any(key.startswith("page.") for key in metadata)


def test_enrich_mutates_the_same_documents_in_place(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    loader = WebBaseLoader("https://example.com/agents")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakePage())
    docs = loader.load()
    doc, acquired_at = docs[0], docs[0].metadata["timestamp.acquired_at"]

    enriched = loader.enrich(docs)

    assert enriched is docs and enriched[0] is doc
    assert doc.metadata["page.title"] == "Agents"
    assert doc.metadata["timestamp.acquired_at"] == acquired_at


def test_enrich_twice_leaves_metadata_unchanged(monkeypatch):
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    loader = WebBaseLoader("https://example.com/agents")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakePage())
    docs = loader.enrich(loader.load())
    once = dict(docs[0].metadata)

    loader.enrich(docs)

    assert docs[0].metadata == once


def test_enrich_leaves_documents_from_another_loader_untouched(monkeypatch):
    from langchain_core.documents import Document
    from velari_ai.integrations.langchain.loaders.loader import WebBaseLoader

    loader = WebBaseLoader("https://example.com/agents")
    monkeypatch.setattr(loader.client, "get", lambda url, **kwargs: _FakePage())
    loader.load()
    foreign = Document(page_content="Churn notes.", metadata={"source.info.uri": "https://other.com/notes"})

    loader.enrich([foreign])

    assert foreign.metadata == {"source.info.uri": "https://other.com/notes"}


def test_auto_loader_enrich_updates_web_documents_and_leaves_file_documents(monkeypatch, tmp_path):
    from velari_core.core.services.client import HttpClient
    from velari_ai.integrations.langchain.loaders.loader import AutoLoader

    monkeypatch.setattr(HttpClient, "get", lambda self, url, **kwargs: _FakePage())
    notes = tmp_path / "notes.txt"
    notes.write_text("Churn fell in Q3.")
    loader = AutoLoader(["https://example.com/agents", str(notes)])
    docs = loader.load()
    file_metadata = dict(docs[1].metadata)

    loader.enrich(docs)

    assert docs[0].metadata["page.title"] == "Agents"
    assert docs[1].metadata == file_metadata


def test_load_gives_table_rows_core_metadata_and_a_row_index_only(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import LocalFileLoader

    path = tmp_path / "orders.csv"
    path.write_text("order_id,status\nORD-1,shipped\nORD-2,returned\n")

    docs = LocalFileLoader(str(path)).load()

    assert [d.metadata["record.index"] for d in docs] == [0, 1]
    assert "content.metrics.length" in docs[0].metadata
    assert not any(key.startswith(("source.table.", "record.fields.")) for key in docs[0].metadata)


def test_file_enrich_mutates_the_same_documents_in_place(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import LocalFileLoader

    path = tmp_path / "orders.csv"
    path.write_text("order_id,status\nORD-1,shipped\nORD-2,returned\n")
    loader = LocalFileLoader(str(path))
    docs = loader.load()
    doc, acquired_at = docs[1], docs[1].metadata["timestamp.acquired_at"]

    enriched = loader.enrich(docs)

    assert enriched is docs and enriched[1] is doc
    assert doc.metadata["record.fields.order_id"] == "ORD-2"
    assert doc.metadata["record.index"] == 1 and doc.metadata["timestamp.acquired_at"] == acquired_at


def test_file_enrich_twice_leaves_metadata_unchanged(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import LocalFileLoader

    path = tmp_path / "orders.csv"
    path.write_text("order_id,status\nORD-1,shipped\n")
    loader = LocalFileLoader(str(path))
    docs = loader.enrich(loader.load())
    once = dict(docs[0].metadata)

    loader.enrich(docs)

    assert docs[0].metadata == once


def test_file_enrich_skips_non_tabular_documents(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import LocalFileLoader

    notes = tmp_path / "notes.txt"
    notes.write_text("Churn fell in Q3.")
    loader = LocalFileLoader(str(notes))
    docs = loader.load()
    before = dict(docs[0].metadata)

    loader.enrich(docs)

    assert docs[0].metadata == before


def test_table_row_without_text_is_valid_but_empty_text_file_is_not():
    from langchain_core.documents import Document
    from velari_ai.integrations.langchain.loaders.loader import DocumentLoader

    assert DocumentLoader.validate([Document(page_content="", metadata={"record.index": 0})])
    assert not DocumentLoader.validate([Document(page_content="", metadata={"source.info.uri": "notes.txt"})])



def test_source_info_name_is_file_stem_for_files_and_last_segment_or_host_for_pages(tmp_path):
    from velari_ai.integrations.langchain.loaders.loader import DocumentLoader
    from velari_ai.integrations.langchain.loaders.metadata import FileMetadata

    def name_of(uri, is_remote):
        metadata = FileMetadata()
        DocumentLoader()._describe(metadata, "Agent notes.", uri=uri, relative="", is_remote=is_remote)
        return metadata.source.info.name

    assert name_of("/srv/data/Order History.csv", False) == "Order History"
    assert name_of("https://example.com/posts/2023-06-23-agent/", True) == "2023-06-23-agent"
    assert name_of("https://example.com/", True) == "example.com"
