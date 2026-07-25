"""
文档向量化与入库脚本。

功能：
加载 raw_docs 目录下的文档，按指定的 chunking 策略切分，
计算 Embedding 后按集合（Collection）独立存储至 Chroma 向量数据库中，
便于针对不同分块策略进行索引和检索效果评估。

用法:
    python src/ingest.py --strategy semantic --collection langchain_semantic
    python src/ingest.py --strategy fixed --collection langchain_fixed
    python src/ingest.py --strategy recursive --collection langchain_recursive
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chunking import ChunkConfig, get_chunks, load_raw_docs

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(_PROJECT_ROOT / "data" / "chroma_db")


def get_embedding_model():
    """获取 Embedding 模型实例。"""
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest(strategy: str, collection_name: str, chunk_size: int, chunk_overlap: int):
    print("加载原始文档...")
    docs = load_raw_docs()
    print(f"共加载 {len(docs)} 篇文档")

    print("加载 Embedding 模型...")
    embedding_model = get_embedding_model()

    print(f"应用策略 [{strategy}] 进行文档切分并写入 Collection [{collection_name}]...")
    config = ChunkConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    t0 = time.time()
    chunks = get_chunks(strategy, docs, config, embedding_model=embedding_model)
    t1 = time.time()
    print(f"文档切分完成: 共 {len(chunks)} 个 Chunk，耗时 {t1 - t0:.1f}s")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    t2 = time.time()
    print(f"向量化及入库完成，耗时 {t2 - t1:.1f}s")
    print(
        f"\n入库完成！Collection: [{collection_name}]，共计 {len(chunks)} 个 Chunk，"
        f"数据持久化至: {CHROMA_PERSIST_DIR}"
    )

    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="文档向量化与 Chroma 入库工具")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["fixed", "recursive", "semantic"],
        help="文档切分策略",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Chroma Collection 名称，建议根据策略做区分",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk 大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk 重叠大小")
    args = parser.parse_args()

    ingest(args.strategy, args.collection, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()