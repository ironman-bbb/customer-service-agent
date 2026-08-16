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
        self.assertTrue(all("可申请" in item.content or "需红冲" in item.content for item in first))
        self.assertTrue(any("# 退款" in item.content for item in first))
        self.assertTrue(any("## 发票" in item.content for item in first))

    def test_split_markdown_recursively_splits_long_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.md"
            path.write_text("# 长文档\n\n" + "这是一条很长的政策说明。" * 20, encoding="utf-8")
            chunks = split_markdown(path, chunk_size=80, chunk_overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(item.content) <= 80 for item in chunks))
        self.assertTrue(all("政策说明" in item.content for item in chunks))


if __name__ == "__main__":
    unittest.main()
