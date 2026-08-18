from __future__ import annotations

from    dataclasses import dataclass, field, fields
from    typing import Any, Dict, Optional, Self, Union

from    omegaconf import DictConfig
from    pydantic import SecretStr
from    langchain_core.embeddings import Embeddings
from    langchain_openai import OpenAIEmbeddings
from    langchain_huggingface import HuggingFaceEmbeddings

# package modules
from    velari_core.core import read_cache_dir
from    ...ai.types import ProviderName

_DEFAULT_MODELS: Dict[ProviderName, str] = {
    ProviderName.OPENAI:                "text-embedding-3-small",
    ProviderName.HUGGINGFACE:           "sentence-transformers/all-mpnet-base-v2",
    ProviderName.SENTENCE_TRANSFORMERS: "sentence-transformers/all-mpnet-base-v2",
}


@dataclass
class ProviderEmbeddingFactory:
    """Resolve provider:model configuration into a langchain `Embeddings` instance.

    Args:
        provider (ProviderName): Which embeddings backend to build; defaults to `ProviderName.OPENAI`.
        model (Optional[str]): Provider-specific model name; defaults to a sensible per-provider
            model (`text-embedding-3-small` for OpenAI, `sentence-transformers/all-mpnet-base-v2`
            for HuggingFace/SentenceTransformers) when omitted.
        extra (Dict[str, Any]): Provider-specific constructor kwargs — e.g. `api_key`/`dimensions`
            for OpenAI. For HuggingFace, `cache_folder` defaults to `read_cache_dir(author="akamlani",
            app="huggingface/hub")`, `encode_kwargs` defaults to `{"normalize_embeddings": True}`
            (required for correct cosine search against `ChromaVectorStorage`'s default
            `distance_metric="cosine"`), and `model_kwargs` defaults to `{"trust_remote_code": False}`
            (blocks a HF model repo from executing custom code at load time unless explicitly opted
            in) when not given. `device` (inside `model_kwargs`) is intentionally left undefaulted —
            `SentenceTransformer` already auto-detects `cuda`/`mps`/`cpu` on its own.

    Examples:
        >>> factory = ProviderEmbeddingFactory(
        ...     provider=ProviderName.HUGGINGFACE, model="sentence-transformers/all-MiniLM-L6-v2",
        ... )
        >>> embeddings = factory.build()
        >>> vectorstore = ChromaVectorStorage(embedding_fn=embeddings).load()
    """
    provider: ProviderName   = field(default=ProviderName.OPENAI)
    model:    Optional[str]  = field(default=None)
    extra:    Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, entry: Union[DictConfig, Dict[str, Any]]) -> Self:
        """Build a ProviderEmbeddingFactory from a raw `embedding_config:`-style mapping.

        Args:
            entry (Union[DictConfig, Dict[str, Any]]): Raw config mapping — `provider`/`model`
                are known fields; anything else (`api_key`, `dimensions`, `cache_folder`, ...)
                lands in `extra`.

        Returns:
            Self: Ready for `.build()`.

        Examples:
            >>> factory = ProviderEmbeddingFactory.from_config({"provider": ProviderName.OPENAI, "dimensions": 512})
            >>> embeddings = factory.build()
        """
        known  = {f.name for f in fields(cls) if f.name != "extra"}
        kwargs = {str(k): v for k, v in entry.items() if k in known}
        extra  = {str(k): v for k, v in entry.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def get_config(self) -> Dict[str, Any]:
        """Resolve `provider`/`model`/`extra` into a plain config dict — no `Embeddings` built.

        For cases that need the resolved configuration (model name plus provider defaults) to
        hand to some external, non-langchain class directly — e.g. a raw
        `sentence_transformers.SentenceTransformer` — without paying for or depending on a
        langchain `Embeddings` wrapper. Unlike `build()`, this never raises: a provider with no
        known defaults (or no `build()` support at all) still gets back `{"model": ..., **extra}`
        unchanged.

        Returns:
            Dict[str, Any]: Always includes `"model"`. For `OPENAI`, a raw string `api_key` is
                wrapped in `SecretStr`. For `HUGGINGFACE`/`SENTENCE_TRANSFORMERS`, `cache_folder`/
                `encode_kwargs`/`model_kwargs` get the same defaults `build()` applies.

        Examples:
            >>> factory = ProviderEmbeddingFactory(provider=ProviderName.HUGGINGFACE)
            >>> config = factory.get_config()
            >>> model = SentenceTransformer(
            ...     config["model"], cache_folder=config["cache_folder"], **config["model_kwargs"],
            ... )
        """
        model = self.model or _DEFAULT_MODELS.get(self.provider, "text-embedding-3-small")
        config: Dict[str, Any] = {"model": model, **self.extra}
        match self.provider:
            case ProviderName.OPENAI:
                if isinstance(config.get("api_key"), str):
                    config["api_key"] = SecretStr(config["api_key"])
            case ProviderName.HUGGINGFACE | ProviderName.SENTENCE_TRANSFORMERS:
                config.setdefault("cache_folder", read_cache_dir(author="akamlani", app="huggingface/hub"))
                config["encode_kwargs"] = {"normalize_embeddings": True, **config.get("encode_kwargs", {})}
                config["model_kwargs"]  = {"trust_remote_code": False, **config.get("model_kwargs", {})}
            case _:
                pass  # no known defaults for this provider — model + extra, unchanged
        return config

    def build(self) -> Embeddings:
        """Construct the concrete `Embeddings` instance for `self.provider`.

        Returns:
            Embeddings: `OpenAIEmbeddings` or `HuggingFaceEmbeddings`, per `self.provider`.

        Raises:
            ValueError: If `self.provider` has no supported embeddings integration.

        Examples:
            >>> factory = ProviderEmbeddingFactory(provider=ProviderName.OPENAI, model="text-embedding-3-small")
            >>> embeddings = factory.build()
            >>> vectorstore = ChromaVectorStorage(embedding_fn=embeddings, collection_name="research").load()
        """
        config = self.get_config()
        model  = config.pop("model")
        match self.provider:
            case ProviderName.OPENAI:
                return OpenAIEmbeddings(model=model, **config)
            case ProviderName.HUGGINGFACE | ProviderName.SENTENCE_TRANSFORMERS:
                return HuggingFaceEmbeddings(model_name=model, **config)
            case _:
                raise ValueError(f"ProviderEmbeddingFactory does not support provider {self.provider!r}")
