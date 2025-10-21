"""Common UI helpers shared across tabs."""
from __future__ import annotations

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


def estimate_tokens(text: str) -> int:
    """Rudimentary token estimate: Chinese chars + latin-word count."""

    import re

    try:
        cjk = len(re.findall(r"[\u4E00-\u9FFF]", text or ""))
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", text or ""))
        return cjk + latin_words
    except Exception as error:  # pragma: no cover - defensive
        report_exception("令牌估算失败", error, level="warning")
        return max(1, len(text or "") // 2)


__all__ = ["estimate_tokens", "report_exception", "stream_text"]
