"""
LLM 客户端封装模块。

基于 OpenAI SDK 兼容接口，支持通过环境变量 LLM_PROVIDER
无缝切换不同的 LLM 提供商（如 OpenAI、DeepSeek 等）。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(self, provider: str | None = None):
        provider = provider or os.getenv("LLM_PROVIDER", "deepseek")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        else:
            raise ValueError(f"未知的 LLM_PROVIDER: {provider}")

        if not api_key:
            raise ValueError(
                f"未找到 {provider} 的 API Key，请检查 .env 文件是否配置正确。"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider = provider

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        model: str | None = None,
    ) -> str:
        """发送 Chat Completion 请求并返回生成的字符串结果"""
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


_client_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """获取单例 LLMClient 实例"""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton