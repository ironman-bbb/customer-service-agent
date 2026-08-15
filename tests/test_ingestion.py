import tempfile
import unittest
from pathlib import Path

from customer_service_agent.ingestion import split_markdown


class IngestionTests(unittest.TestCase):
    def test_split_markdown_keeps_source_and_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.md"
            path.write_text("# 退款\n\n7 天内可申请。\n\n## 发票\n\n专票需红冲。", encoding="utf-8")
            first = split_markdown(path)
            second = split_markdown(path)

        self.assertTrue(first)
        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])
        self.assertTrue(all(item.source == "policy.md" for item in first))


if __name__ == "__main__":
    unittest.main()

