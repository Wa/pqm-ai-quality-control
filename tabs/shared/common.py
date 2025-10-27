"""Common UI helpers shared across tabs."""
from __future__ import annotations

import os
import time
from typing import Any

try:  # pragma: no cover - optional dependency during backend execution
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - backend or tests without Streamlit
    st = None  # type: ignore


def report_exception(message: str, error: Exception, *, level: str = "error") -> None:
    """Log exceptions to Streamlit when available, otherwise fallback to stderr."""

    formatted = f"{message}: {error}"
    if st is not None:
        log_fn = getattr(st, level, None)
        if callable(log_fn):
            log_fn(formatted)
            return
        # Fall back to Streamlit error display if requested level missing
        st.error(formatted)
    else:  # pragma: no cover - lightweight fallback for backend workers
        print(formatted)


def stream_text(
    placeholder: Any,
    text: str,
    *,
    chunk_size: int = 30,
    render_method: str = "text",
    delay: float | None = None,
) -> None:
    """Stream text into a Streamlit placeholder in word-sized chunks."""

    words = (text or "").split()
    if not words:
        method = getattr(placeholder, render_method, None)
        if callable(method):
            method("")
        else:
            placeholder.write("")
        return

    method = getattr(placeholder, render_method, None)
    if not callable(method):
        method = placeholder.write

    buffered_words: list[str] = []
    for start in range(0, len(words), chunk_size):
        buffered_words.append(" ".join(words[start : start + chunk_size]))
        method(" ".join(buffered_words).strip())
        if delay and delay > 0:
            time.sleep(delay)


_TOKENIZER_CACHE: dict[str, object | bool] = {}


def _resolve_tokenizer_name(model_hint: str | None) -> str | None:
    if not model_hint:
        return None
    base = model_hint.split(":", 1)[0].strip().lower()
    if not base:
        return None
    if "qwen" in base:
        return "qwen3"
    if base in {"gpt_oss", "gpt-oss", "gptoss"}:
        return "gpt_oss"
    return base.replace("-", "_")


def estimate_tokens(text: str, model_hint: str | None = None):
    """Return a measured token count when possible, otherwise fall back."""

    text = text or ""
    if not text:
        return 0

    tokenizer_name = _resolve_tokenizer_name(model_hint)
    if tokenizer_name:
        cached = _TOKENIZER_CACHE.get(tokenizer_name)
        if cached is False:
            return f"{tokenizer_name} model missing"
        if cached is None:
            model_path = tokenizer_name
            if not os.path.isdir(model_path):
                _TOKENIZER_CACHE[tokenizer_name] = False
                return f"{tokenizer_name} model missing"
            try:
                from transformers import AutoTokenizer  # type: ignore

                _TOKENIZER_CACHE[tokenizer_name] = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
            except Exception as error:  # pragma: no cover - heavy dependency issues
                report_exception("加载分词器失败", error, level="warning")
                _TOKENIZER_CACHE[tokenizer_name] = False
                return f"{tokenizer_name} model missing"
        tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
        if tokenizer and tokenizer is not True:
            try:
                return len(tokenizer.encode(text))  # type: ignore[call-arg]
            except Exception as error:  # pragma: no cover - tokenizer runtime failure
                report_exception("分词器编码失败", error, level="warning")

    import re

    try:
        cjk = len(re.findall(r"[\u4E00-\u9FFF]", text))
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
        return cjk + latin_words
    except Exception as error:  # pragma: no cover - defensive
        report_exception("令牌估算失败", error, level="warning")
        return max(1, len(text) // 2)


__all__ = ["estimate_tokens", "report_exception", "stream_text"]
