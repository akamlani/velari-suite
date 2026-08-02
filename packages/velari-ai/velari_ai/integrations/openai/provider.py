from __future__ import annotations
from    typing import List, Union, overload
from    omegaconf import DictConfig
# specific modules
from    openai import OpenAI
# package modules
from    ...ai.provider import ProviderEmbeddings

class OpenAIProviderEmbeddings(ProviderEmbeddings):
    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._embedding_model = embedding_model

    @classmethod
    def from_config(cls, cfg: DictConfig, api_key: str) -> OpenAIProviderEmbeddings:
        return cls(
            api_key=api_key,
            embedding_model=cfg.embedding,
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
        )

    @overload
    def get_embeddings(self, texts: str) -> List[float]: ...
    @overload
    def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...
    def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        inputs = [texts] if isinstance(texts, str) else texts
        response = self._client.embeddings.create(model=self._embedding_model, input=inputs)
        embeddings = [data.embedding for data in response.data]
        return embeddings[0] if isinstance(texts, str) else embeddings
