"""
DeepSeek LLM 客户端模块

本模块提供与 DeepSeek API 的交互接口。
特点：
- 仅用于语义决策，不生成统计代码
- 兼容 OpenAI API 格式
- 支持图片输入（用于图表理解）

注意：API 密钥应通过环境变量配置，不要硬编码在源码中。
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class LLMError(RuntimeError):
    """LLM 调用错误异常"""
    pass


@dataclass(slots=True)
class DeepSeekClient:
    """DeepSeek API 客户端"""
    
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip() or None
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", self.base_url).rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", self.model)

    @property
    def configured(self) -> bool:
        """检查是否已配置 API 密钥"""
        return bool(self.api_key)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        images: Iterable[str] = (),
        model: str | None = None,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """调用 LLM 生成 JSON 响应"""
        if not self.api_key:
            raise LLMError("DEEPSEEK_API_KEY is not configured")
        user_content: str | list[dict[str, Any]] = user
        image_parts = [self._image_part(path) for path in images if Path(path).exists()]
        if image_parts:
            user_content = [{"type": "text", "text": user}, *image_parts]
        payload = {
            "model": model or self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        }
        try:
            return self._request(payload)
        except LLMError as first_error:
            # Some compatible gateways or older DeepSeek deployments reject
            # response_format even though they can follow a JSON-only prompt.
            if "HTTP 400" not in str(first_error):
                raise
            payload.pop("response_format", None)
            try:
                return self._request(payload)
            except LLMError as second_error:
                raise LLMError(f"DeepSeek JSON request failed after compatibility retry: {second_error}") from first_error

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            content = response_data["choices"][0]["message"]["content"]
            return self._parse_json(content)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                detail = ""
            raise LLMError(f"HTTP {exc.code} from DeepSeek: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError, LLMError) as exc:
            raise LLMError(f"DeepSeek request failed: {exc}") from exc

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise LLMError("DeepSeek response must be a JSON object")
        return value

    @staticmethod
    def _image_part(path: str) -> dict[str, Any]:
        suffix = Path(path).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}