import unittest

from pydantic import ValidationError

from customer_service_agent.schemas import Order


class SchemaTests(unittest.TestCase):
    def test_invalid_usage_rate_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Order(
                order_id="ORDER-1000",
                user_id="USER-X",
                product="基础版",
                amount=1,
                days_since_purchase=1,
                usage_rate=1.1,
                invoice_status="未开票",
            )


if __name__ == "__main__":
    unittest.main()

