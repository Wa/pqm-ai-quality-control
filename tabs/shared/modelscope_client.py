"""Shared ModelScope client for chat completions (non-streaming)."""
from __future__ import annotations

from typing import Dict, List, Optional

import requests


class ModelScopeClient:
    """Minimal HTTP client for invoking ModelScope chat completions.

    Designed to be shared across tabs (special symbols, ai agent, etc.).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api-inference.modelscope.cn/v1",
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> None:
        self._session = requests.Session()
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout

    def chat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, object]]] = None,
        stream: bool = False,
        options: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if stream:
            raise NotImplementedError("ModelScopeClient does not support streaming responses")

        target_model = model or self._model
        payload: Dict[str, object] = {
            "model": target_model,
            "messages": messages or [],
            "temperature": self._temperature,
            "stream": False,
            "enable_thinking": False,  # Required for some models (e.g., Qwen3-32B) in non-streaming mode
        }

        num_ctx: Optional[int] = None
        if options and isinstance(options, dict):
            num_ctx_value = options.get("num_ctx")
            if isinstance(num_ctx_value, int):
                num_ctx = num_ctx_value
        payload["max_context_length"] = num_ctx or 40001

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response = self._session.post(url, json=payload, headers=headers, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()

        message_content = ""
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        message_content = message["content"]
                        break
                    if isinstance(choice.get("text"), str):
                        message_content = str(choice["text"])
                        break
            if not message_content and isinstance(data.get("text"), str):
                message_content = str(data["text"])
            if not message_content:
                output_text = data.get("output_text") or data.get("output")
                if isinstance(output_text, str):
                    message_content = output_text
            if not message_content:
                data_field = data.get("data")
                if isinstance(data_field, list) and data_field:
                    first_item = data_field[0]
                    if isinstance(first_item, dict):
                        for key in ("text", "output_text", "content"):
                            value = first_item.get(key)
                            if isinstance(value, str) and value:
                                message_content = value
                                break

        if not message_content:
            message_content = ""

        return {
            "message": {"role": "assistant", "content": message_content},
            "model": target_model,
            "provider": "modelscope",
            "stats": {},
        }


