"""Pydantic 数据模型。

与 C# DTO 类似，但 Pydantic 会在运行时校验外部输入。
"""

from enum import Enum

from pydantic import BaseModel, Field


class InvoiceStatus(str, Enum):
    NONE = "未开票"
    NORMAL = "普票"
    SPECIAL = "专票"


class Intent(str, Enum):
    GREETING = "问候"
    PRODUCT = "产品咨询"
    ORDER = "订单查询"
    REFUND = "退款申请"
    COMPLAINT = "投诉"
    OTHER = "其他"


class Order(BaseModel):
    order_id: str = Field(pattern=r"^ORDER-\d{4}$")
    user_id: str = Field(min_length=1)
    product: str = Field(min_length=1)
    amount: float = Field(ge=0)
    days_since_purchase: int = Field(ge=0)
    usage_rate: float = Field(ge=0, le=1)
    invoice_status: InvoiceStatus
    is_upgrade: bool = False


class RefundDecision(BaseModel):
    order_id: str
    eligible: bool
    refundable_amount: float = Field(ge=0)
    reasons: list[str]
    requires_human_approval: bool = True


class KnowledgeHit(BaseModel):
    content: str
    source: str
    chunk_id: str
    score: float = Field(ge=-1, le=1)


class CustomerServiceResult(BaseModel):
    intent: Intent
    answer: str
    order_id: str | None = None
    requires_human: bool = False
    recommended_action: str | None = None
    references: list[str] = Field(default_factory=list)
