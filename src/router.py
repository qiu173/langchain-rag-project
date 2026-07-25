"""
查询路由模块 (Query Router)：判断用户输入的查询意图并分发至不同的处理策略。

路由策略划分：
- DIRECT: 直接由 LLM 回答（适用于闲聊、通用常识类问题，避免不必要的检索开销）
- RETRIEVE: 走标准 RAG 单次检索流程（适用于具体的单个文档知识点查询）
- MULTI_HOP: 将复杂复合问题拆解为多个子查询，依次检索后综合生成结果
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from llm_client import get_llm_client


class RouteType(str, Enum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"
    MULTI_HOP = "multi_hop"


@dataclass
class RouteDecision:
    route: RouteType
    sub_queries: list[str] | None = None  # 仅在 MULTI_HOP 策略下生效
    reasoning: str = ""


ROUTER_PROMPT = """你是一个查询路由器，负责判断用户问题应该如何处理。

判断规则：
1. DIRECT：问题是闲聊、常识性问题，或者是关于 LangChain 是什么/由谁开发这类
   不需要查文档细节的问题
2. RETRIEVE：问题需要查询 LangChain 文档中的具体信息（API用法、概念解释、
   使用示例等），但是单一、聚焦的问题
3. MULTI_HOP：问题复杂，涉及多个概念/模块的综合（比如"如何用 LCEL 实现一个
   带记忆的多轮对话链，并支持流式输出"这种需要综合多个文档片段的问题），
   需要拆解成多个子问题

请以严格的 JSON 格式输出，不要包含任何额外的解析文字：
{{"route": "direct|retrieve|multi_hop", "sub_queries": ["子问题1", "子问题2"] 或 null, "reasoning": "简短说明"}}

用户问题：{query}
"""


def route_query(query: str) -> RouteDecision:
    client = get_llm_client()
    prompt = ROUTER_PROMPT.format(query=query)

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # 规范化清理 LLM 返回的 Markdown 代码块标记（如 ```json ... ```）
    cleaned_response = response.strip()
    if cleaned_response.startswith("```"):
        lines = cleaned_response.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_response = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned_response)
        return RouteDecision(
            route=RouteType(parsed["route"]),
            sub_queries=parsed.get("sub_queries"),
            reasoning=parsed.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # 路由解析失败时的兜底策略：回退到标准 RETRIEVE 检索，保证系统稳定性
        print(f"[router] 路由解析失败，降级回退至 RETRIEVE。原始输出: {response!r}, 异常原因: {e}")
        return RouteDecision(
            route=RouteType.RETRIEVE,
            reasoning="路由结果解析失败，触发系统兜底策略",
        )


if __name__ == "__main__":
    test_queries = [
        "你好",
        "LangChain 是什么时候发布的",
        "如何用 RecursiveCharacterTextSplitter 切分文本",
        "如何用 LCEL 实现一个带记忆的多轮对话链，并且支持流式输出，同时能调用外部工具",
    ]
    for q in test_queries:
        decision = route_query(q)
        print(f"Q: {q}")
        print(f"  -> route={decision.route.value}, sub_queries={decision.sub_queries}")
        print(f"  reasoning: {decision.reasoning}\n")