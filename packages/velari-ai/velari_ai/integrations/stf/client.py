from __future__ import annotations
from    typing                 import Any, List
from    sentence_transformers import SentenceTransformer
# specific modules
from    velari_core.core               import read_cache_dir
from    velari_core.core.perf.profiler import profile_time
# package modules
from    ...ai import types
from    ...ai.schemas.response import PerfMetrics, UsageMetrics


class SentenceTransformerClient(object):
    """Owns everything specific to a local `sentence_transformers.SentenceTransformer` model —
    construction (with the same `cache_folder`/`trust_remote_code` defaults as
    `ProviderEmbeddingFactory.get_config()`'s HuggingFace branch in
    `integrations/langchain/models/provider.py`, without importing that module — it imports
    `langchain_core`/`langchain_openai`/`langchain_huggingface` at module scope, which would
    pull LangChain in as a hard dependency for a plain local embedding call) and embedding,
    returning a plain `types.EmbeddingResult` rather than a raw tensor/ndarray.

    Args:
        model_name (str): Hub id or local path; sets the speed/quality trade-off and vector size.
            Both are general-purpose English models, so pick by size and speed:
            `all-MiniLM-L6-v2` — 384-dim, 256-token limit, ~23M params; smaller and faster, good for prototyping.
            `all-mpnet-base-v2` — 768-dim, 384-token limit, ~109M params; better quality, slower, larger vectors.
            Vectors from different models aren't comparable — index and query with the same one.
        **kwargs (Any): Forwarded to `SentenceTransformer()`, merged over the `cache_folder`/
            `trust_remote_code` defaults (caller-supplied values win).

    Examples:
        >>> client = SentenceTransformerClient("all-mpnet-base-v2")
        >>> result = client.embed(["How do I reset my billing password?", "Update payment method"])
        >>> len(result.result.embeddings[0])
        768
    """
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        client_kwargs = {
            "cache_folder": read_cache_dir(author="akamlani", app="huggingface/hub"),
            "trust_remote_code": False,
            **kwargs,
        }
        self._model = SentenceTransformer(model_name, **client_kwargs)

    @profile_time(target=lambda result: result.metrics.perf)
    def embed(self, texts: List[str], **kwargs: Any) -> types.EmbeddingResult:
        vectors = self._model.encode(texts, normalize_embeddings=True, **kwargs)
        # No billing API to report usage, unlike OpenAI — count real (non-padding) tokens via
        # the model's own tokenizer so usage isn't a fabricated 0; embeddings have no output and
        # no separate input/output split, so input_tokens mirrors total_tokens.
        total_tokens = int(self._model.preprocess(texts)["attention_mask"].sum().item())
        return types.EmbeddingResult(
            result=types.EmbeddingVectors(embeddings=vectors.tolist()),
            metrics=types.ProviderMetrics(
                perf=PerfMetrics(latency_sec=0.0),
                usage=UsageMetrics(input_tokens=total_tokens, total_tokens=total_tokens),
            ),
        )
