"""Markdown 文档加载、切块与向量化入库。"""

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .embeddings import EmbeddingService


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    source: str
    content: str


def split_markdown(path: Path, max_chars: int = 1200) -> list[DocumentChunk]:
    """保留 Markdown 标题语义，并将过长段落按字符数继续切分。"""

    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=#{1,6}\s)|\n\s*\n", text)
    pieces: list[str] = []
    current_heading = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if section.startswith("#"):
            lines = section.splitlines()
            current_heading = lines[0]
        content = f"{current_heading}\n{section}" if current_heading not in section else section
        for start in range(0, len(content), max_chars):
            piece = content[start : start + max_chars].strip()
            if piece:
                pieces.append(piece)

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

