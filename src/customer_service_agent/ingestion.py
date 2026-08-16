"""Markdown 文档加载、切块与向量化入库。"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .embeddings import EmbeddingService


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    source: str
    content: str


def split_markdown(
    path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    """先按 Markdown 标题分组，再递归切分过长章节。"""

    text = path.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "一级标题"),
            ("##", "二级标题"),
            ("###", "三级标题"),
            ("####", "四级标题"),
            ("#####", "五级标题"),
            ("######", "六级标题"),
        ],
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    documents = recursive_splitter.split_documents(sections)
    pieces = [
        document.page_content.strip()
        for document in documents
        if any(
            line.strip() and not line.lstrip().startswith("#")
            for line in document.page_content.splitlines()
        )
    ]

    chunks: list[DocumentChunk] = []
    for index, content in enumerate(pieces):
        raw_id = f"{path.name}:{index}:{content}".encode()
        chunks.append(
            DocumentChunk(
                chunk_id=sha256(raw_id).hexdigest(),
                source=path.name,
                content=content,
            )
        )
    return chunks


def load_knowledge(directory: Path) -> list[DocumentChunk]:
    return [
        chunk
        for path in sorted(directory.glob("*.md"))
        for chunk in split_markdown(path)
    ]


def vectorize_chunks(
    chunks: list[DocumentChunk], embeddings: EmbeddingService, batch_size: int = 32
) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embeddings.embed_documents([chunk.content for chunk in batch])
        rows.extend(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "content": chunk.content,
                "vector": vector,
            }
            for chunk, vector in zip(batch, vectors, strict=True)
        )
    return rows
