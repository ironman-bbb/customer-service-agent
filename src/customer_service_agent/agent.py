"""项目中唯一的 Agent：真实 LangChain Agent。"""

from typing import Any

from .config import Settings
from .prompts import SYSTEM_PROMPT
from .schemas import CustomerServiceResult


def create_customer_service_agent(
    settings: Settings, tools: list[Any], checkpointer: Any
) -> Any:
    try:
        from langchain.agents import create_agent
        from langchain.agents.structured_output import ToolStrategy
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError("请先安装项目依赖：python -m pip install -e .") from error

    model = ChatOpenAI(
        model=settings.chat_model_name,
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        timeout=60,
        max_retries=2,
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        # 选用 ToolStrategy，兼容更多实现 Tool Calling 的 OpenAI 兼容模型。
        response_format=ToolStrategy(CustomerServiceResult),
        checkpointer=checkpointer,
    )
