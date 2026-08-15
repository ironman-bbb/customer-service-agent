import tempfile
import unittest
from pathlib import Path

from customer_service_agent.repositories import OrderNotFoundError, OrderRepository


class OrderRepositoryTests(unittest.TestCase):
    def test_reads_order_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orders.json"
            path.write_text(
                """[
                    {
                      "order_id":"ORDER-1001", "user_id":"USER-1", "product":"专业版",
                      "amount":1999, "days_since_purchase":3, "usage_rate":0.2,
                      "invoice_status":"未开票", "is_upgrade":false
                    }
                ]""",
                encoding="utf-8",
            )
            repository = OrderRepository.from_json(path)

        self.assertEqual(repository.get(" order-1001 ").product, "专业版")

    def test_unknown_order_is_not_invented(self) -> None:
        repository = OrderRepository([])
        with self.assertRaises(OrderNotFoundError):
            repository.get("ORDER-9999")


if __name__ == "__main__":
    unittest.main()

