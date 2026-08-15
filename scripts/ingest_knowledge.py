"""调用 Embedding API，将 knowledge/*.md 向量化并写入 Milvus。"""

import argparse

from dotenv import load_dotenv

from customer_service_agent.config import PROJECT_ROOT, Settings
from customer_service_agent.embeddings import EmbeddingService
from customer_service_agent.ingestion import load_knowledge, vectorize_chunks
from customer_service_agent.retrieval import MilvusKnowledgeStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除并重建指定 Milvus collection（更换向量维度时使用）",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings.from_environment()
    embeddings = EmbeddingService(settings)
    store = MilvusKnowledgeStore(settings, embeddings)
    if args.recreate:
        store.recreate_collection()
    else:
        store.ensure_collection()

    chunks = load_knowledge(PROJECT_ROOT / "knowledge")
    rows = vectorize_chunks(chunks, embeddings)
    count = store.upsert(rows)
    print(f"向 Milvus 写入 {count} 个知识切块。")


if __name__ == "__main__":
    main()

