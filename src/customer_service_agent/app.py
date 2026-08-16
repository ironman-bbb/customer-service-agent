"""唯一启动入口：真实模型 + PostgreSQL 记忆 + Milvus RAG。"""

import json
import sys
from typing import Any

from .agent import create_customer_service_agent
from .config import PROJECT_ROOT, Settings
from .embeddings import EmbeddingService
from .repositories import OrderRepository
from .retrieval import MilvusKnowledgeStore
from .schemas import CustomerServiceResult
from .tools import build_tools


def _content(message: Any) -> str:
    value = getattr(message, "content", "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("请先安装项目依赖：python -m pip install -e .") from error

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings.from_environment()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as error:
        raise RuntimeError("请先安装项目依赖：python -m pip install -e .") from error

    orders = OrderRepository.from_json(settings.orders_path)
    embedding_service = EmbeddingService(settings)
    knowledge = MilvusKnowledgeStore(settings, embedding_service)
    with PostgresSaver.from_conn_string(settings.postgres_uri) as checkpointer:
        checkpointer.serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("customer_service_agent.schemas", "CustomerServiceResult"),
                ("customer_service_agent.schemas", "Intent"),
            ]
        )
        # 首次执行会创建 LangGraph 自己的 checkpoint 表。
        checkpointer.setup()
        agent = create_customer_service_agent(
            settings, build_tools(orders, knowledge), checkpointer
        )

        thread_id = (
            input("会话 ID（直接回车使用 local-demo）：").strip()
            or "local-demo"
        )
        config = {"configurable": {"thread_id": thread_id}}
        print("客服 Agent 已启动。输入 /clear 清空当前记忆，quit 退出。")

        while True:
            user_text = input("\n你：").strip()
            if user_text.lower() in {"quit", "exit", "q"}:
                break
            if user_text == "/clear":
                checkpointer.delete_thread(thread_id)
                print("已清空当前会话记忆。")
                continue
            if not user_text:
                continue

            try:
                # Checkpointer 会根据 thread_id 自动读取历史、并保存每个 Agent 步骤。
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config=config,
                )
                structured = result.get("structured_response")
                if isinstance(structured, CustomerServiceResult):
                    print("客服：" + structured.answer)
                    print(
                        "结构化结果："
                        + json.dumps(
                            structured.model_dump(mode="json"), ensure_ascii=False
                        )
                    )
                else:
                    print("客服：" + _content(result["messages"][-1]))
            # CLI 边界需将模型、Tool、PostgreSQL 或 Milvus 的异常转为可读提示。
            except Exception as error:  # noqa: BLE001
                print(f"调用失败：{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
