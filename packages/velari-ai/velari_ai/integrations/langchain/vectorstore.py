import   hashlib
import   logging
import   pandas as pd
from     typing import Any, Optional, List, Dict, Tuple, Union, Sequence, Self
# specific modules
from     langchain_core.documents import Document
from     langchain_chroma import Chroma
# package modules
from    ...ai.retrieval.vectorstore import RetrieverStrategy, VectorStore

logger = logging.getLogger(__name__)

class LangChainVectorStorage(VectorStore):
    def __init__(self, embedding_fn: Any, collection_name: str = "default") -> None:
        super().__init__()
        self._embedding_fn:    Any           = embedding_fn
        self._collection_name: str           = collection_name
        self._client:          Any           = None
        self._vectorstore:     Optional[Any] = None

    def exists(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
            return True
        except Exception:
            return False

    def is_empty(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
        except Exception:
            return True
        return len(self) == 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._vectorstore = None

    def _normalize_dicts(
        self,
        documents:  List[dict],
        text_field: Optional[str],
    ) -> Tuple[List[str], List[dict]]:
        if text_field is None:
            raise ValueError("text_field is required when documents is List[dict]")
        texts = [doc[text_field] for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k != text_field} for doc in documents]
        return texts, metadatas

    def to_frame(
        self,
        candidates: Sequence[Union[Document, Tuple[Document, float]]],
    ) -> pd.DataFrame:
        """Flatten retrieve_candidates() results into a single DataFrame — one row per candidate.

        Args:
            candidates (Sequence[Union[Document, Tuple[Document, float]]]): Output of
                `retrieve_candidates()` — a bare `Document`, or `(Document, score)` when the
                strategy returns one (e.g. `VECTORSTORE_DISTANCE_SCORE`).

        Returns:
            pd.DataFrame: `id`, `page_content`, one `meta.<key>` column per metadata key, and a
                `score` column when the candidates carry one.

        Examples:
            >>> store = ChromaVectorStorage(embedding_fn=embeddings).load()
            >>> candidates = store.retrieve_candidates(
            ...     "agent memory types", strategy=RetrieverStrategy.VECTORSTORE_DISTANCE_SCORE,
            ... )
            >>> df_candidates = store.to_frame(candidates)
            >>> df_candidates[["id", "meta.source", "score"]]
        """
        rows = []
        for candidate in candidates:
            doc, score = candidate if isinstance(candidate, tuple) else (candidate, None)
            row = {
                "id":           doc.id,
                "page_content": doc.page_content,
                **{f"meta.{k}": v for k, v in doc.metadata.items()},
            }
            if score is not None:
                row["score"] = score
            rows.append(row)
        return pd.DataFrame(rows)


class ChromaVectorStorage(LangChainVectorStorage):
    def __init__(
        self,
        embedding_fn:      Any,
        collection_name:   str           = "default",
        persist_directory: Optional[str] = None,
        distance_metric:   str           = "cosine",
    ) -> None:
        import chromadb
        super().__init__(embedding_fn, collection_name)
        self._persist_dir     = persist_directory
        self._distance_metric = distance_metric
        self._client           = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory else
            chromadb.EphemeralClient()
        )

    @property
    def _collection_kwargs(self) -> Dict[str, Any]:
        return {
            "client":              self._client,
            "collection_name":     self._collection_name,
            "collection_metadata": {"hnsw:space": self._distance_metric},
        }

    def __len__(self) -> int:
        if self._vectorstore is None:
            return 0
        return self._vectorstore._collection.count()

    def load(self) -> Self:
        self._vectorstore = Chroma(embedding_function=self._embedding_fn, **self._collection_kwargs)
        return self


    def _require_vectorstore(self) -> Any:
        if self._vectorstore is None:
            raise RuntimeError("Vector store not loaded — call load() first")
        return self._vectorstore

    def retrieve_candidates(
        self,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.VECTORSTORE_SIMILARITY,
        k: int = 3,
        **kwargs: Any,
    ) -> Sequence[Union[Document, Tuple[Document, float]]]:
        """Retrieve candidates for `query` via the given strategy.

        Args:
            query (str): The search query.
            strategy (RetrieverStrategy): Which Chroma search method to use.
            k (int): Number of results to return.
            **kwargs (Any): Forwarded to the underlying method — strategy-specific (e.g.
                `fetch_k`/`lambda_mult` for `VECTORSTORE_MMR`; `filter`/`where_document`
                for the direct-search strategies).

        Returns:
            Sequence[Union[Document, Tuple[Document, float]]]: `Document`s, or
                `(Document, score)` pairs for the score-returning strategies.
        """
        try:
            vectorstore = self._require_vectorstore()
            if strategy == RetrieverStrategy.VECTORSTORE_DISTANCE_SCORE:
                return vectorstore.similarity_search_with_score(query, k=k, **kwargs)
            if strategy == RetrieverStrategy.VECTORSTORE_RELEVANCE_SCORE:
                return vectorstore.similarity_search_with_relevance_scores(query, k=k, **kwargs)
            if strategy == RetrieverStrategy.VECTORSTORE_MMR:
                mmr_defaults = {"fetch_k": 20, "lambda_mult": 0.5}
                return vectorstore.max_marginal_relevance_search(query, k=k, **{**mmr_defaults, **kwargs})
            if strategy == RetrieverStrategy.VECTORSTORE_RETRIEVER:
                return vectorstore.as_retriever(search_kwargs={"k": k, **kwargs}).invoke(query)
            return vectorstore.similarity_search(query, k=k, **kwargs)
        except RuntimeError as e:
            logger.error(f"Vector store not loaded — call load() first: {e}")
            return []

    def create_index(self,
        documents: List[Document],
        text_field: str,
        batch_size: int
    ) -> None:
        texts, metadatas = self._extract_texts_and_metadatas(documents, text_field)
        embeddings = self._embedding_fn.embed_documents(texts)
        self.upsert_merge(texts, metadatas, embeddings, batch_size)

    def update(
        self,
        ids: List[str],
        documents:  Optional[List[str]] = None,
        metadatas:  Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        if self._vectorstore is None:
            return
        if documents is not None and embeddings is None:
            embeddings = self._embedding_fn.embed_documents(documents)
        kwargs: Dict[str, Any] = {"ids": ids}
        if documents is not None:
            kwargs["documents"]  = documents
        if metadatas is not None:
            kwargs["metadatas"]  = metadatas
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._vectorstore._collection.update(**kwargs)

    def upsert_merge(
        self,
        texts:      List[str],
        metadatas:  List[dict],
        embeddings: List[List[float]],
        batch_size: int,
    ) -> None:
        ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        if self._vectorstore is None:
            self._vectorstore = Chroma(embedding_function=self._embedding_fn, **self._collection_kwargs)
        self._vectorstore._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def upsert_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 100,
    ) -> None:
        """Upsert already-computed embeddings directly, bypassing embedding_fn recomputation."""
        self.upsert_merge(texts, metadatas or [{} for _ in texts], embeddings, batch_size)

    def get_embeddings(self, ids: List[str]) -> List[List[float]]:
        if self._vectorstore is None:
            return []
        result = self._vectorstore._collection.get(ids=ids, include=["embeddings"])
        return result["embeddings"]

    def get_all_embeddings(self) -> List[List[float]]:
        if self._vectorstore is None:
            return []
        result = self._vectorstore._collection.get(include=["embeddings"])
        return result["embeddings"]

    def _extract_texts_and_metadatas(
        self,
        documents: Union[List[dict], List[Document]],
        text_field: Optional[str],
    ) -> Tuple[List[str], List[dict]]:
        if documents and isinstance(documents[0], Document):
            texts = [doc.page_content for doc in documents if isinstance(doc, Document)]
            metadatas = [doc.metadata for doc in documents if isinstance(doc, Document)]
            return texts, metadatas
        dicts = [doc for doc in documents if isinstance(doc, dict)]
        return self._normalize_dicts(dicts, text_field)

    def _from_documents(self, documents: List[Document], batch_size: int) -> Any:
        return Chroma.from_documents(
            documents=documents,
            batch_size=batch_size,
            embedding=self._embedding_fn,
            **self._collection_kwargs,
        )

    def _from_texts(self, texts: List[str], metadatas: List[dict], batch_size: int) -> Any:
        # texts, metadatas = self._normalize_dicts(documents, text_field)
        # self._vectorstore = self._from_texts(texts, metadatas, batch_size)
        return Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            batch_size=batch_size,
            embedding=self._embedding_fn,
            **self._collection_kwargs,
        )
