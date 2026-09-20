from    typing import Any
# package modules
from    ..base import EmbeddingAdapter
from    ...schemas.requests import GatewayEmbeddingRequest
from    ...schemas.response import GatewayEmbeddingResponse
from    ......ai.types import ModelConfig, ProviderName
from    ......ai.schemas.response import PerfMetrics
from    ......integrations.stf.client import SentenceTransformerClient

_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


class SentenceTransformersEmbeddingAdapter(EmbeddingAdapter):
    """Maps `GatewayEmbeddingRequest`/`GatewayEmbeddingResponse` to/from `SentenceTransformerClient`
    — all local-model request/response handling lives in `integrations/stf/client.py`. Used for
    both `ProviderName.HUGGINGFACE` and `ProviderName.SENTENCE_TRANSFORMERS` (synonymous, per
    `integrations/langchain/models/provider.py`'s `ProviderEmbeddingFactory`).
    """
    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(model_config, **kwargs)
        model_name = self._model_config.model or _DEFAULT_MODEL
        self._client = SentenceTransformerClient(model_name, **self._client_kwargs)

    def embed_texts(self, request: GatewayEmbeddingRequest) -> GatewayEmbeddingResponse:
        result = self._client.embed(request.texts, **request.parameters)
        return GatewayEmbeddingResponse(
            model=GatewayEmbeddingResponse.Model(provider=ProviderName.SENTENCE_TRANSFORMERS, model=self._model_config.model),
            metrics=GatewayEmbeddingResponse.Metrics(
                perf=PerfMetrics(latency_sec=round(result.metrics.perf.latency_sec, 3)),
                usage=result.metrics.usage,
            ),
            result=GatewayEmbeddingResponse.Result(embeddings=result.result.embeddings),
        )
