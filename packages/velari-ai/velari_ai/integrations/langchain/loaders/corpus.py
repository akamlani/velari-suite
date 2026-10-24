from __future__ import annotations

import  re
import  pandas as pd
from    functools import cached_property
from    typing import List, Optional, Self

from    langchain_core.documents import Document

# package modules
from    velari_data.info import InfoText
from    ...openai.tokenizer import Tokenizer
from    .loader import DocumentLoader


class DocumentCorpus(object):
    """Load documents from any `DocumentLoader` and expose them as a per-document frame, size stats, and a context string.

    Call `load()` once (core metadata), optionally `enrich()` (further metadata, in place); the frame
    properties are computed lazily from the loaded documents and cached.

    Args:
        loader (DocumentLoader): `WebBaseLoader`, `LocalFileLoader`, or any other loader to read from.
        name (Optional[str]): Identifier for this corpus, e.g. its vector collection name; see `name`.
        model (str): Model whose tiktoken encoding counts tokens; defaults to `"gpt-4o-mini"`.

    Examples:
        >>> loader = LocalFileLoader("data/cs/datasets/faq.csv", text_field="question")
        >>> corpus = DocumentCorpus(loader, name="support-faq").load().enrich()
        >>> corpus.name
        'support-faq'
        >>> corpus.df_docs[["meta.record.fields.category", "meta.content.metrics.token_sz"]]
        >>> corpus.df_metrics_stats.loc[["sum", "mean"], "meta.content.metrics.token_sz"]
    """
    def __init__(self, loader: DocumentLoader, name: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self._loader = loader
        self._name = name
        self._tokenizer = Tokenizer(model)
        self._docs: Optional[List[Document]] = None

    def load(self) -> Self:
        """Read every source from the loader, so calls chain: `DocumentCorpus(loader).load().df_docs`.

        Raises:
            ValueError: If no documents were loaded or any has neither text nor a table-row index.
        """
        docs = self._loader.load()
        if not DocumentLoader.validate(docs):
            raise ValueError(f"{type(self._loader).__name__} returned no documents, or one with neither text nor a table-row index")
        self._docs = docs
        return self

    def enrich(self) -> Self:
        """Add further metadata (e.g. a web page's `page.*`) to the loaded documents in place, so calls chain.

        Cached frames are dropped so they are rebuilt with the new metadata on next access.

        Examples:
            >>> corpus = DocumentCorpus(WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")).load()
            >>> corpus.enrich().df_docs[["meta.page.title", "meta.page.language"]]
        """
        self._loader.enrich(self.docs)
        for name, attr in vars(type(self)).items():
            if isinstance(attr, cached_property):
                self.__dict__.pop(name, None)
        return self

    @property
    def docs(self) -> List[Document]:
        """The loaded `Document`s; raises `RuntimeError` until `load()` has been called."""
        if self._docs is None:
            raise RuntimeError("Corpus not loaded — call load() first")
        return self._docs

    @property
    def name(self) -> str:
        """The `name` given, else a slug of the single source loaded's `source.info.name` (file stem, or page's last URL segment or host).

        Raises:
            RuntimeError: If no `name` was given and `load()` hasn't been called.
            ValueError: If no `name` was given and the documents come from more than one source.
        """
        if self._name is not None:
            return self._name.strip()
        (name,) = {doc.metadata["source.info.name"] for doc in self.docs}  # ValueError unless exactly one source
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    @property
    def context(self) -> str:
        """All `page_content` joined into one string, for building prompt context."""
        return DocumentLoader.to_string(self.docs)

    @cached_property
    def df_docs(self) -> pd.DataFrame:
        """One row per document: `page_content`, `meta.*` (no `meta.page.links`), and the `meta.content.metrics.*` sizes."""
        df_raw = DocumentLoader.to_frame(self.docs)
        df_sizes = InfoText.calc_record_stats(self._tokenizer.pipe(df_raw, "page_content"), "page_content")
        df_sizes = df_sizes.filter(regex=r"^\w+_sz$").add_prefix("meta.content.metrics.")
        return df_raw.drop(columns=["meta.page.links"], errors="ignore").join(df_sizes)

    @cached_property
    def df_metadata(self) -> pd.DataFrame:
        """The `meta.*` columns of `df_docs`."""
        return self.df_docs.filter(like="meta.")

    @cached_property
    def df_content(self) -> pd.DataFrame:
        """`page_content` and the other content-related columns of `df_docs` (preview, metrics)."""
        return self.df_docs.filter(like="content")

    @cached_property
    def df_metrics_stats(self) -> pd.DataFrame:
        """Sum/min/max/std/mean/median/skew/kurtosis of the `meta.content.metrics.*_sz` size columns."""
        return InfoText.calc_frame_stats(self.df_docs.filter(regex=r"^meta\.content\.metrics\..+_sz$"))
