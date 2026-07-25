"""
端到端 RAG 流水线 (Pipeline) 实现。

整合查询路由 (Query Routing)、向量检索 (Retrieval)、重排序 (Reranking) 及生成 (Generation)。
默认参数配置:
- collection_name: "langchain_semantic"
- use_rerank: True
(该配置为基于评估实验分析选出的最优检索方案组合)

用法示例:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    answer = pipeline.answer("如何用 LCEL 实现流式输出？")
    print(answer.text)
    print(answer.sources)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from llm_client import get_llm_client
from retriever import Retriever
from router import RouteType, route_query

load_dotenv()

ANSWER_PROMPT = """你是 LangChain 文档助手，请基于下面提供的文档片段回答用户问题。
如果文档片段中没有足够信息回答问题，请明确说明，不要编造。

文档片段：
{context}

用户问题：{query}

请给出准确、简洁的回答，如果涉及代码用法，给出示例代码。
"""


@dataclass
class Answer:
    text: str
    sources: list[str]
    route: str


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "langchain_semantic",
        use_rerank: bool = True,
        top_k: int = 5,
        embedding_model_name: str | None = None,
    ):
        embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.retriever = Retriever(
            collection_name=collection_name,
            embedding_model=self.embedding_model,
            use_rerank=use_rerank,
        )

        self.llm = get_llm_client()
        self.top_k = top_k

    def _generate(self, query: str, context_docs: list) -> str:
        context = "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in context_docs
        )

        prompt = ANSWER_PROMPT.format(
            context=context,
            query=query,
        )

        return self.llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

    def answer(self, query: str) -> Answer:
        decision = route_query(query)

        if decision.route == RouteType.DIRECT:
            text = self.llm.chat(
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            )
            return Answer(text=text, sources=[], route="direct")

        elif decision.route == RouteType.RETRIEVE:
            result = self.retriever.retrieve(query, top_k=self.top_k)

            text = self._generate(query, result.documents)
            sources = [
                d.metadata.get("source", "未知") for d in result.documents
            ]

            return Answer(
                text=text,
                sources=sources,
                route="retrieve",
            )

        elif decision.route == RouteType.MULTI_HOP:
            sub_queries = decision.sub_queries or [query]

            all_docs = []

            for sq in sub_queries:
                result = self.retriever.retrieve(
                    sq,
                    top_k=max(2, self.top_k // len(sub_queries)),
                )
                all_docs.extend(result.documents)

            # 根据 source 和前 50 个字符进行文档去重
            seen = set()
            unique_docs = []

            for d in all_docs:
                key = (
                    d.metadata.get("source"),
                    d.page_content[:50],
                )

                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

            text = self._generate(query, unique_docs)

            sources = [
                d.metadata.get("source", "未知") for d in unique_docs
            ]

            return Answer(
                text=text,
                sources=sources,
                route="multi_hop",
            )

        else:
            raise ValueError(f"未知的路由类型: {decision.route}")


if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_qs = [
        "你好",
        "如果模型返回了两个 tool_calls，但我只提供了一个 ToolMessage，会引发什么错误？",
    ]

    for q in test_qs:
        print(f"\nQ: {q}")

        ans = pipeline.answer(q)

        print(f"路由: {ans.route}")
        print(f"回答: {ans.text}")

        if ans.sources:
            print(f"引用来源: {ans.sources}")