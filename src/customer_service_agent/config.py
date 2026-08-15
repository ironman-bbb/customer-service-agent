"""集中读取对话模型、Embedding、PostgreSQL 和 Milvus 配置。"""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"请在 .env 中配置 {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    chat_model_name: str
    chat_api_key: str
    chat_base_url: str

    embedding_model_name: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_dimensions: int

    postgres_base_url: str
    postgres_database: str

    milvus_uri: str
    milvus_token: str
    milvus_database: str
    milvus_collection: str

    orders_path: Path = PROJECT_ROOT / "data" / "orders.json"
    retrieval_top_k: int = 4

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            chat_model_name=_required("CHAT_MODEL_NAME"),
            chat_api_key=_required("CHAT_API_KEY"),
            chat_base_url=_required("CHAT_BASE_URL"),
            embedding_model_name=_required("EMBEDDING_MODEL_NAME"),
            embedding_api_key=_required("EMBEDDING_API_KEY"),
            embedding_base_url=_required("EMBEDDING_BASE_URL"),
            embedding_dimensions=int(_required("EMBEDDING_DIMENSIONS")),
            postgres_base_url=_required("DB_URL"),
            postgres_database=_required("POSTGRES_DATABASE"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530").strip(),
            milvus_token=os.getenv("MILVUS_TOKEN", "").strip(),
            milvus_database=os.getenv("MILVUS_DATABASE", "default").strip(),
            milvus_collection=os.getenv(
                "MILVUS_COLLECTION", "customer_service_knowledge"
            ).strip(),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
        )

    @property
    def postgres_uri(self) -> str:
        """将 DB_URL 的数据库路径替换为显式配置的 POSTGRES_DATABASE。"""

        parsed = urlsplit(self.postgres_base_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.netloc:
            raise RuntimeError("DB_URL 必须是合法的 postgres:// 或 postgresql:// 地址")
        database_path = "/" + quote(self.postgres_database, safe="")
        return urlunsplit(
            (parsed.scheme, parsed.netloc, database_path, parsed.query, parsed.fragment)
        )
