"""Agent 可调用的业务 Tools。"""

from typing import Any

from .repositories import OrderRepository
from .retrieval import MilvusKnowledgeStore
from .schemas import InvoiceStatus, Order, RefundDecision


def evaluate_refund_order(order: Order) -> RefundDecision:
    """纯业务函数：无数据库、无模型，因此容易单元测试。"""

    reasons: list[str] = []
    if order.days_since_purchase > 7:
        reasons.append("超过支付后 7 个自然日")
    if order.usage_rate > 0.5:
        reasons.append("AI 问答额度使用量超过 50%")
    if order.is_upgrade:
        reasons.append("升级订单不适用首购退款")

    eligible = not reasons
    if eligible:
        reasons.append("订单符合基本退款条件")
    if order.invoice_status is InvoiceStatus.SPECIAL:
        reasons.append("已开专票，须先完成红字发票流程")

    return RefundDecision(
        order_id=order.order_id,
        eligible=eligible,
        refundable_amount=order.amount if eligible else 0,
        reasons=reasons,
        requires_human_approval=True,
    )


def build_tools(
    orders: OrderRepository, knowledge: MilvusKnowledgeStore
) -> list[Any]:
    """将已测试的 Python 业务能力包装成 LangChain Tools。"""

    try:
        from langchain.tools import tool
    except ImportError as error:
        raise RuntimeError("请先安装项目依赖：python -m pip install -e .") from error

    @tool
    def query_order(order_id: str) -> dict:
        """当用户查询 ORDER-xxxx 订单的产品、金额、使用量或发票时使用。"""

        return orders.get(order_id).model_dump(mode="json")

    @tool
    def evaluate_refund(order_id: str) -> dict:
        """当用户询问指定订单是否符合退款条件时使用；只审核，不执行退款。"""

        return evaluate_refund_order(orders.get(order_id)).model_dump(mode="json")

    @tool
    def search_knowledge(question: str) -> list[dict]:
        """当用户询问产品、价格、退款、发票或 FAQ 时检索 Milvus 知识库。"""

        return [hit.model_dump() for hit in knowledge.search(question)]

    return [query_order, evaluate_refund, search_knowledge]

