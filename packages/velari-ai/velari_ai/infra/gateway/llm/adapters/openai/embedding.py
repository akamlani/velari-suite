from    typing import Any
# package modules
from    ..base import EmbeddingAdapter
from    ...schemas.requests import GatewayEmbeddingRequest
from    ...schemas.response import GatewayEmbeddingResponse
from    ......ai.types import ModelConfig, ProviderName
from    ......ai.schemas.response import PerfMetrics
from    ......integrations.openai.client import OpenAIClient


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Maps `GatewayEmbeddingRequest`/`GatewayEmbeddingResponse` to/from `OpenAIClient` — all
    request-building and response-parsing lives in `integrations/openai/client.py`.
    """
    def __init__(self, model_config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(model_config, **kwargs)
        self._client = OpenAIClient(**self._client_kwargs)

    def embed_texts(self, request: GatewayEmbeddingRequest) -> GatewayEmbeddingResponse:
        result = self._client.create_embeddings(model=self._model_config.model, texts=request.texts, **request.parameters)
        return GatewayEmbeddingResponse(
            model=GatewayEmbeddingResponse.Model(provider=ProviderName.OPENAI, model=self._model_config.model),
            metrics=GatewayEmbeddingResponse.Metrics(
                perf=PerfMetrics(latency_sec=round(result.metrics.perf.latency_sec, 3)),
                usage=result.metrics.usage,
            ),
            result=GatewayEmbeddingResponse.Result(embeddings=result.result.embeddings),
        )
