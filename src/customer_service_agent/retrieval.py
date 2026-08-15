"""Milvus 向量入库与语义检索。"""

from typing import Any

from .config import Settings
from .embeddings import EmbeddingService
from .schemas import KnowledgeHit


class MilvusKnowledgeStore:
    def __init__(self, settings: Settings, embeddings: EmbeddingService) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as error:
            raise RuntimeError("请先安装项目依赖：python -m pip install -e .") from error

        client_kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            client_kwargs["token"] = settings.milvus_token
        if settings.milvus_database:
            client_kwargs["db_name"] = settings.milvus_database

        self._client = MilvusClient(**client_kwargs)
        self._collection = settings.milvus_collection
        self._dimensions = settings.embedding_dimensions
        self._top_k = settings.retrieval_top_k
        self._embeddings = embeddings

    def ensure_collection(self) -> None:
        if self._client.has_collection(collection_name=self._collection):
            return

        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._dimensions,
        )
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded",
        )

    def recreate_collection(self) -> None:
        if self._client.has_collection(collection_name=self._collection):
            self._client.drop_collection(collection_name=self._collection)
        self.ensure_collection()

    def upsert(self, rows: list[dict]) -> int:
        self.ensure_collection()
        if not rows:
            return 0
        result = self._client.upsert(collection_name=self._collection, data=rows)
        return int(result.get("upsert_count", len(rows)))

    def search(self, question: str) -> list[KnowledgeHit]:
        self.ensure_collection()
        vector = self._embeddings.embed_query(question)
        result = self._client.search(
            collection_name=self._collection,
            data=[vector],
            anns_field="vector",
            limit=self._top_k,
            output_fields=["chunk_id", "source", "content"],
            search_params={"metric_type": "COSINE", "params": {}},
        )
        if not result:
            return []
        return [
            KnowledgeHit(
                content=hit["entity"]["content"],
                source=hit["entity"]["source"],
                # pymilvus 3.x 以实际主键名返回，而不是旧版常见的 `id`。
                chunk_id=hit["entity"]["chunk_id"],
                score=float(hit["distance"]),
            )
            for hit in result[0]
        ]
