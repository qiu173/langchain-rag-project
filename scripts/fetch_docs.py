"""
数据采集脚本：拉取 LangChain 官方文档源文件（markdown/mdx），并锁定到指定 commit。

说明：
LangChain 官方文档维护于独立仓库 langchain-ai/docs（Mintlify 构建）。
脚本会克隆指定分支/commit，将文档抽取并清洗至 data/raw_docs/ 目录下，
去除前端组件语法（如 <Tabs>、import 语句等），以减少后续分块与检索中的噪声。

用法:
    python scripts/fetch_docs.py --ref main
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/langchain-ai/docs.git"
DOCS_SUBDIR = "src/oss"

CLONE_DIR = Path("data/_langchain_repo")
OUTPUT_DIR = Path("data/raw_docs")


def clone_repo(ref: str) -> str:
    """克隆仓库并返回当前的 commit hash。"""
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 克隆 langchain-ai/docs 仓库 (ref={ref})...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            REPO_URL,
            str(CLONE_DIR),
        ],
        check=True,
    )

    commit_hash = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CLONE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    print(f"当前 commit hash: {commit_hash}")
    return commit_hash


def clean_mdx_content(text: str) -> str:
    """去除 MDX / Mintlify 特有语法，仅保留可读正文。"""
    # 去掉 import 语句
    text = re.sub(r"^import .*$", "", text, flags=re.MULTILINE)

    # 去掉自闭合标签（如 <Icon .../>）
    text = re.sub(r"<Icon\b[^>]*/?>", "", text)

    # 去掉成对出现的组件标签，保留标签内部的正文内容
    component_tags = (
        "Tabs",
        "TabItem",
        "CodeBlock",
        "Admonition",
        "Tip",
        "Note",
        "Warning",
        "Info",
        "Danger",
        "Check",
        "CodeGroup",
        "Accordion",
        "AccordionGroup",
        "Card",
        "CardGroup",
        "Frame",
        "Steps",
        "Step",
        "ParamField",
        "ResponseField",
        "Expandable",
    )
    tag_pattern = "|".join(component_tags)
    text = re.sub(rf"</?(?:{tag_pattern})[^>]*>", "", text)

    # 去掉容器语法（如 :::python / :::）
    text = re.sub(r"^:::\w*\s*$", "", text, flags=re.MULTILINE)

    # 去掉 Frontmatter（--- ... ---）
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # 压缩连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_docs(include_dirs: list[str] | None = None):
    """提取并清洗 Markdown/MDX 文件。"""
    src_dir = CLONE_DIR / DOCS_SUBDIR
    if not src_dir.exists():
        raise FileNotFoundError(
            f"未找到文档目录 {src_dir}，请检查 DOCS_SUBDIR 路径是否正确。"
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    md_files = list(src_dir.rglob("*.md")) + list(src_dir.rglob("*.mdx"))

    if include_dirs:
        md_files = [
            f
            for f in md_files
            if any(part in include_dirs for part in f.relative_to(src_dir).parts)
        ]

    print(f"[2/3] 共找到 {len(md_files)} 个文档文件，开始清洗...")

    count = 0
    for f in md_files:
        try:
            raw = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        cleaned = clean_mdx_content(raw)
        if len(cleaned) < 50:
            continue

        rel_path = f.relative_to(src_dir)
        out_path = OUTPUT_DIR / rel_path.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")
        count += 1

    print(f"[3/3] 成功清洗并保存 {count} 份文档至 {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="拉取并清洗 LangChain 官方文档")
    parser.add_argument(
        "--ref",
        default="main",
        help="指定 langchain-ai/docs 仓库的分支或 tag (默认: main)",
    )
    parser.add_argument(
        "--keep-repo",
        action="store_true",
        help="保留克隆的完整原始仓库",
    )
    parser.add_argument(
        "--include-dirs",
        nargs="+",
        default=["langchain"],
        help="指定包含的子目录列表 (默认只保留 'langchain')",
    )
    args = parser.parse_args()

    commit_hash = clone_repo(args.ref)
    include_dirs = args.include_dirs if args.include_dirs != [""] else None
    collect_docs(include_dirs=include_dirs)

    if not args.keep_repo:
        shutil.rmtree(CLONE_DIR)
        print("已清理临时仓库，仅保留清洗后的数据。")

    print("\n数据处理完成。拉取信息摘要：")
    print(f"  Repo: langchain-ai/docs")
    print(f"  Ref: {args.ref}")
    print(f"  Commit: {commit_hash}")


if __name__ == "__main__":
    main()