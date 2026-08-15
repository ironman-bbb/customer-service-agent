"""JSON 订单数据访问。

后期可以保持 `get()` 接口不变，将本实现替换为 PostgreSQL 版。
"""

import json
from pathlib import Path

from .schemas import Order


class OrderNotFoundError(LookupError):
    pass


class OrderRepository:
    def __init__(self, orders: list[Order]) -> None:
        self._orders = {order.order_id: order for order in orders}

    @classmethod
    def from_json(cls, path: Path) -> "OrderRepository":
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        return cls([Order.model_validate(item) for item in raw_items])

    def get(self, order_id: str) -> Order:
        normalized = order_id.strip().upper()
        try:
            return self._orders[normalized]
        except KeyError as error:
            raise OrderNotFoundError(f"订单 {normalized} 不存在") from error

