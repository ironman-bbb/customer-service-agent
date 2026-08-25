"""独立的 Embedding 模型适配层。"""

from .config import Settings
from langchain_openai import OpenAIEmbeddings

class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._dimensions = settings.embedding_dimensions
        self._client = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            max_retries=2,
            request_timeout=60,
            check_embedding_ctx_length=False,
            model_kwargs={
               "encoding_format": "float",
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._client.embed_documents(texts)
        self._validate(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._client.embed_query(text)
        self._validate([vector])
        return vector

    def _validate(self, vectors: list[list[float]]) -> None:
        wrong = [len(vector) for vector in vectors if len(vector) != self._dimensions]
        if wrong:
            raise ValueError(
                f"Embedding 实际维度 {wrong[0]} 与 EMBEDDING_DIMENSIONS="
                f"{self._dimensions} 不一致"
            )
