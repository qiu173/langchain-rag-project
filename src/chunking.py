"""
文本分块（Chunking）策略实现模块。

本模块包含三种用于 RAG 系统的文档切分策略：
1. Fixed: 固定长度切分（Baseline）
2. Recursive: 基于 Markdown 标题结构与递归字符的联合切分
3. Semantic: 基于相邻句子 Embedding 相似度衰减的语义切分

用于不同切分策略在检索效果上的对比实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 锚定项目根目录，确保路径解析一致
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW_DIR = str(_PROJECT_ROOT / "data" / "raw_docs")

ChunkStrategy = Literal["fixed", "recursive", "semantic"]


@dataclass
class ChunkConfig:
    strategy: ChunkStrategy
    chunk_size: int = 500
    chunk_overlap: int = 50


def load_raw_docs(raw_dir: str = _DEFAULT_RAW_DIR) -> list[Document]:
    """加载指定目录下的 Markdown 文档，并在 metadata 中记录相对路径。"""
    docs = []
    for path in Path(raw_dir).rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path.relative_to(raw_dir))},
            )
        )
    return docs


def chunk_fixed(docs: list[Document], config: ChunkConfig) -> list[Document]:
    """策略一：固定长度分块 (Baseline)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def chunk_recursive(docs: list[Document], config: ChunkConfig) -> list[Document]:
    """策略二：递归结构化分块 (基于 Markdown 标题及递归字符)"""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    all_chunks = []
    for doc in docs:
        header_chunks = header_splitter.split_text(doc.page_content)
        for hc in header_chunks:
            hc.metadata.update(doc.metadata)
        sub_chunks = char_splitter.split_documents(header_chunks)
        all_chunks.extend(sub_chunks)
    return all_chunks


def chunk_semantic(
    docs: list[Document], config: ChunkConfig, embedding_model=None
) -> list[Document]:
    """策略三：语义分块 (依赖 Embedding 模型计算相邻句相似度)"""
    from langchain_experimental.text_splitter import SemanticChunker

    if embedding_model is None:
        raise ValueError("语义分块模式下需要传入 embedding_model 实例")

    splitter = SemanticChunker(embedding_model)
    return splitter.split_documents(docs)


def get_chunks(
    strategy: ChunkStrategy,
    docs: list[Document],
    config: ChunkConfig | None = None,
    embedding_model=None,
) -> list[Document]:
    """根据指定的策略分块文档"""
    config = config or ChunkConfig(strategy=strategy)
    if strategy == "fixed":
        return chunk_fixed(docs, config)
    elif strategy == "recursive":
        return chunk_recursive(docs, config)
    elif strategy == "semantic":
        return chunk_semantic(docs, config, embedding_model)
    else:
        raise ValueError(f"未知的分块策略: {strategy}")


if __name__ == "__main__":
    docs = load_raw_docs()
    print(f"成功加载 {len(docs)} 个原始文档")

    for strategy in ["fixed", "recursive"]:
        chunks = get_chunks(strategy, docs)
        print(f"\n策略 [{strategy}]: 切分生成 {len(chunks)} 个 chunks")
        if chunks:
            sample = chunks[0]
            print(f"示例 Chunk Metadata: {sample.metadata}")
            print(f"示例 Chunk 内容片段: {sample.page_content[:200]}...")