"""
评估脚本：对比不同分块策略及是否开启 Rerank 的检索效果。

用法:
    python eval/evaluate.py --collections langchain_fixed langchain_recursive langchain_semantic --rerank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 锚定到项目根目录，避免运行路径不同导致相对路径解析错误
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from langchain_huggingface import HuggingFaceEmbeddings
from retriever import Retriever

_DEFAULT_TESTSET_PATH = str(_PROJECT_ROOT / "eval" / "testset.json")
_DEFAULT_RESULTS_PATH = _PROJECT_ROOT / "eval" / "results.json"


def load_testset(path: str = _DEFAULT_TESTSET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_sources: list[str], gold_sources: list[str]) -> float | None:
    """计算 Recall@K：检索到的文档来源中，命中了多少比例的标准答案来源。"""
    if not gold_sources:
        return None  # direct 类型问题不参与检索评估
    hit = sum(1 for g in gold_sources if any(g in r for r in retrieved_sources))
    return hit / len(gold_sources)


def mrr(retrieved_sources: list[str], gold_sources: list[str]) -> float | None:
    """计算 MRR (Mean Reciprocal Rank)：第一个命中的标准答案排在第几位，取倒数。"""
    if not gold_sources:
        return None
    for i, r in enumerate(retrieved_sources):
        if any(g in r for g in gold_sources):
            return 1.0 / (i + 1)
    return 0.0


def evaluate_collection(
    collection_name: str,
    testset: list[dict],
    embedding_model,
    use_rerank: bool,
    top_k: int = 5,
) -> dict:
    retriever = Retriever(
        collection_name=collection_name,
        embedding_model=embedding_model,
        use_rerank=use_rerank,
    )

    recalls, mrrs = [], []
    for item in testset:
        if item.get("type") == "direct":
            continue  # 不需要检索的问题跳过检索评估

        result = retriever.retrieve(item["question"], top_k=top_k)
        retrieved_sources = [d.metadata.get("source", "") for d in result.documents]

        r = recall_at_k(retrieved_sources, item["gold_sources"])
        m = mrr(retrieved_sources, item["gold_sources"])
        if r is not None and m is not None:
            recalls.append(r)
            mrrs.append(m)

    return {
        "collection": collection_name,
        "use_rerank": use_rerank,
        "n_questions": len(recalls),
        f"recall@{top_k}": sum(recalls) / len(recalls) if recalls else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
    }


def print_comparison_table(results: list[dict]):
    print("\n" + "=" * 70)
    print("检索效果评估结果")
    print("=" * 70)
    header = f"{'Collection':<28}{'Rerank':<10}{'N':<6}{'Recall@K':<12}{'MRR':<10}"
    print(header)
    print("-" * 70)
    for r in results:
        recall_key = [k for k in r if k.startswith("recall@")][0]
        recall_val = r.get(recall_key)
        recall_str = f"{recall_val:.3f}" if recall_val is not None else "N/A"
        mrr_str = f"{r['mrr']:.3f}" if r["mrr"] is not None else "N/A"
        print(
            f"{r['collection']:<28}{str(r['use_rerank']):<10}"
            f"{r['n_questions']:<6}{recall_str:<12}{mrr_str:<10}"
        )
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="评估不同分块与 Rerank 策略的检索效果")
    parser.add_argument(
        "--collections",
        nargs="+",
        required=True,
        help="要对比的 Chroma collection 名称列表",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="是否对每个 collection 额外评估开启 Rerank 的效果",
    )
    parser.add_argument("--top-k", type=int, default=5, help="检索 Top-K 数量")
    parser.add_argument("--testset", default=_DEFAULT_TESTSET_PATH, help="测试集路径")
    args = parser.parse_args()

    testset = load_testset(args.testset)
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    results = []
    for collection in args.collections:
        results.append(
            evaluate_collection(
                collection,
                testset,
                embedding_model,
                use_rerank=False,
                top_k=args.top_k,
            )
        )
        if args.rerank:
            results.append(
                evaluate_collection(
                    collection,
                    testset,
                    embedding_model,
                    use_rerank=True,
                    top_k=args.top_k,
                )
            )

    print_comparison_table(results)

    out_path = _DEFAULT_RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"评估结果已成功保存到 {out_path}")


if __name__ == "__main__":
    main()