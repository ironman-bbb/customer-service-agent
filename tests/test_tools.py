import unittest

from customer_service_agent.schemas import InvoiceStatus, Order
from customer_service_agent.tools import evaluate_refund_order


def order(**changes) -> Order:
    values = {
        "order_id": "ORDER-1001",
        "user_id": "USER-1",
        "product": "专业版",
        "amount": 1999,
        "days_since_purchase": 3,
        "usage_rate": 0.2,
        "invoice_status": InvoiceStatus.NONE,
        "is_upgrade": False,
    }
    values.update(changes)
    return Order.model_validate(values)


class RefundTests(unittest.TestCase):
    def test_eligible_order(self) -> None:
        result = evaluate_refund_order(order())
        self.assertTrue(result.eligible)
        self.assertEqual(result.refundable_amount, 1999)

    def test_over_seven_days(self) -> None:
        result = evaluate_refund_order(order(days_since_purchase=8))
        self.assertFalse(result.eligible)
        self.assertIn("7 个自然日", result.reasons[0])

    def test_over_half_usage(self) -> None:
        self.assertFalse(evaluate_refund_order(order(usage_rate=0.51)).eligible)

    def test_upgrade_order(self) -> None:
        self.assertFalse(evaluate_refund_order(order(is_upgrade=True)).eligible)


if __name__ == "__main__":
    unittest.main()

