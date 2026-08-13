"""Tests for velari_ai.integrations.langchain.vectorstore."""


def test_is_empty_true_before_collection_exists():
    from velari_ai.integrations.langchain.vectorstore import ChromaVectorStorage

    store = ChromaVectorStorage(embedding_fn=None, collection_name="test-collection")

    assert store.is_empty()


def test_is_empty_true_when_loaded_with_no_documents():
    from velari_ai.integrations.langchain.vectorstore import ChromaVectorStorage

    store = ChromaVectorStorage(embedding_fn=None, collection_name="test-collection")
    store._client.create_collection("test-collection")
    store.load()

    assert store.is_empty()


def test_is_empty_false_after_adding_a_document():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from velari_ai.integrations.langchain.vectorstore import ChromaVectorStorage

    store = ChromaVectorStorage(
        embedding_fn=DeterministicFakeEmbedding(size=8), collection_name="test-collection",
    )
    store.load()

    assert store._vectorstore is not None
    store._vectorstore.add_texts(["hello world"])

    assert not store.is_empty()
