"""
辅助构建测试集脚本。

功能：
从指定的文档目录中随机抽样片段，利用 LLM 生成候选问题，
减少纯手工撰写测试集的工作量。生成结果需经人工确认/修正后，
再合并至最终的测试集文件 (testset.json)。

用法:
    python eval/build_testset.py --n 30
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# 将 src 目录添加到 Python Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import get_llm_client

GEN_PROMPT = """下面是一段 LangChain 文档内容。请基于这段内容生成 1 个用户可能会问的真实问题（不要问"这段文档讲了什么"这种空泛问题，要像真实用户在使用 LangChain 时会问的具体问题）。

文档内容：
{content}

请以 JSON 格式输出，不要有其他文字：
{{"question": "生成的问题"}}
"""


def sample_docs(raw_dir: str, n: int) -> list[Path]:
    """从指定目录随机抽样 markdown 文件"""
    all_files = list(Path(raw_dir).rglob("*.md"))
    return random.sample(all_files, min(n, len(all_files)))


def clean_json_response(response_text: str) -> str:
    """清理 LLM 返回结果中的 Markdown 代码块标记"""
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="生成 RAG 评估测试集候选问题")
    parser.add_argument("--n", type=int, default=30, help="生成候选问题的数量")
    parser.add_argument("--raw-dir", default="data/raw_docs", help="原始文档目录")
    parser.add_argument("--out", default="eval/candidate_questions.json", help="输出候选问题路径")
    args = parser.parse_args()

    client = get_llm_client()
    files = sample_docs(args.raw_dir, args.n)

    candidates = []
    for f in files:
        content = f.read_text(encoding="utf-8")[:1500]  # 截断避免超长
        if len(content) < 100:
            continue

        prompt = GEN_PROMPT.format(content=content)
        response = client.chat(messages=[{"role": "user", "content": prompt}], temperature=0.7)

        try:
            cleaned_text = clean_json_response(response)
            parsed = json.loads(cleaned_text)
            
            candidates.append({
                "question": parsed["question"],
                "source_file": str(f.relative_to(args.raw_dir)),
                "type": "single_doc",  # 默认标记为单文档问题
                "gold_sources": [str(f.relative_to(args.raw_dir))],
                "gold_answer_summary": "",  # 需人工补充
            })
        except (json.JSONDecodeError, KeyError):
            print(f"跳过解析失败的文件: {f}")
            continue

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    print(f"成功生成 {len(candidates)} 条候选问题，已保存至 {args.out}")
    print("提示：请完成人工审核，补充 gold_answer_summary 及补全 multi_hop 问题的 gold_sources，最后合并进 testset.json。")


if __name__ == "__main__":
    main()