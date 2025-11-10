"""MCP-style tool adapters for the AI Agent (filesystem + http + conversion)."""
from __future__ import annotations

import os
import re
import contextlib
import io
import json
import multiprocessing
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests

from tabs.shared.file_conversion import (
    process_pdf_folder,
    process_word_ppt_folder,
    process_excel_folder,
    process_textlike_folder,
    process_archives,
)


def _safe_jsonable(value: Any) -> Any:
    """Best-effort conversion of objects to JSON-serialisable forms."""

    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return json.loads(json.dumps(str(value)))
        except Exception:
            return repr(value)


def _safe_join(base: str, *paths: str) -> str:
    base_abs = os.path.abspath(base)
    candidate = os.path.abspath(os.path.join(base_abs, *paths))
    if not candidate.startswith(base_abs + os.sep) and candidate != base_abs:
        raise PermissionError("Path escapes base directory")
    return candidate


def prepare_conversation_dirs(
    session_dirs: Dict[str, str],
    conversation_id: Optional[str],
) -> Dict[str, str]:
    """Return session dirs scoped to a specific conversation, creating folders if needed."""

    if not conversation_id:
        return dict(session_dirs)

    conv_dirs = dict(session_dirs)

    def _ensure(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        os.makedirs(path, exist_ok=True)
        return path

    uploads_base = session_dirs.get("ai_agent_inputs")
    if uploads_base:
        uploads_path = os.path.join(uploads_base, conversation_id)
        _ensure(uploads_path)
        conv_dirs["ai_agent_inputs"] = uploads_path

    generated_base = session_dirs.get("generated_ai_agent")
    if generated_base:
        generated_root = os.path.join(generated_base, conversation_id)
        _ensure(generated_root)
        conv_dirs["generated_ai_agent"] = generated_root
        for name in ("examined_txt", "initial_results", "final_results", "checkpoint", "logs"):
            sub_path = os.path.join(generated_root, name)
            _ensure(sub_path)
            conv_dirs[f"generated_ai_agent_{name}"] = sub_path
    else:
        for name in ("examined_txt", "initial_results", "final_results", "checkpoint", "logs"):
            key = f"generated_ai_agent_{name}"
            base = session_dirs.get(key)
            if base:
                sub_path = os.path.join(base, conversation_id)
                _ensure(sub_path)
                conv_dirs[key] = sub_path

    return conv_dirs


def get_agent_paths(session_dirs: Dict[str, str]) -> Dict[str, str]:
    return {
        "uploads_inputs": session_dirs.get("ai_agent_inputs", ""),
        "generated_root": session_dirs.get("generated_ai_agent", ""),
        "examined_txt": session_dirs.get("generated_ai_agent_examined_txt", ""),
        "initial_results": session_dirs.get("generated_ai_agent_initial_results", ""),
        "final_results": session_dirs.get("generated_ai_agent_final_results", ""),
        "checkpoint": session_dirs.get("generated_ai_agent_checkpoint", ""),
        "logs": session_dirs.get("generated_ai_agent_logs", ""),
    }


def tool_filesystem(
    action: str,
    path: str,
    *,
    session_dirs: Dict[str, str],
    content: Optional[str] = None,
) -> Dict[str, object]:
    """Restricted filesystem operations under this session's ai_agent roots."""

    paths = get_agent_paths(session_dirs)
    allowed_roots = [p for p in (paths["uploads_inputs"], paths["generated_root"]) if p]
    if not allowed_roots:
        raise RuntimeError("AI agent directories not initialized")

    # Determine root by longest prefix match
    target = None
    for root in sorted(allowed_roots, key=len, reverse=True):
        if os.path.isabs(path) and os.path.abspath(path).startswith(os.path.abspath(root)):
            target = path
            break
        try:
            target = _safe_join(root, path)
            break
        except Exception:
            continue
    if not target:
        raise PermissionError("Path not under allowed roots")

    if action == "read_text":
        with open(target, "r", encoding="utf-8", errors="ignore") as handle:
            return {"path": target, "text": handle.read()}
    if action == "write_text":
        if content is None:
            raise ValueError("content required for write_text")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
        return {"path": target, "written": len(content)}
    if action == "list":
        if os.path.isdir(target):
            return {
                "path": target,
                "files": sorted([name for name in os.listdir(target)]),
            }
        return {"path": target, "files": []}

    raise ValueError(f"Unsupported filesystem action: {action}")


def tool_http_fetch(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 2_000_000,
    allowed_domains: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Restricted HTTP GET for planning/execution steps."""

    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("Only http/https URLs are allowed")
    if allowed_domains:
        import urllib.parse as _url

        host = _url.urlparse(url).hostname or ""
        if host not in set(allowed_domains):
            raise PermissionError("Domain not allowed")

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content[: max_bytes]
    text = None
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("utf-8", errors="ignore")
    return {"url": url, "status": resp.status_code, "text": text}


def tool_web_search(
    query: str,
    *,
    max_results: int = 5,
    timeout: float = 15.0,
) -> Dict[str, object]:
    """Perform a lightweight DuckDuckGo instant-answer search."""

    if not query or not query.strip():
        raise ValueError("query is required")

    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "t": "pqm-ai-agent",
    }
    url = "https://duckduckgo.com/"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict[str, str]] = []

    def _extract(items: List[Dict[str, object]]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            if "FirstURL" in item and "Text" in item:
                results.append({
                    "title": str(item.get("Text", "")),
                    "url": str(item.get("FirstURL", "")),
                })
            if "Topics" in item and isinstance(item.get("Topics"), list):
                _extract(item["Topics"])  # type: ignore[arg-type]

    related = data.get("RelatedTopics") or []
    if isinstance(related, list):
        _extract(related)  # type: ignore[arg-type]

    abstract_text = data.get("AbstractText")
    abstract_url = data.get("AbstractURL")
    if abstract_text and abstract_url:
        results.insert(0, {"title": str(abstract_text), "url": str(abstract_url)})

    return {
        "query": query,
        "results": results[: max(1, max_results)],
        "source": "duckduckgo",
    }


def tool_python_exec(
    code: str,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Dict[str, object]:
    """Execute Python code in a restricted subprocess sandbox."""

    if not code or not code.strip():
        raise ValueError("code is required")

    inputs = inputs or {}
    queue: multiprocessing.Queue = multiprocessing.Queue()

    def _worker(payload: str, params: Dict[str, Any], out_queue: multiprocessing.Queue) -> None:
        allowed_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "sorted": sorted,
            "round": round,
        }
        stdout_capture = io.StringIO()
        locals_env: Dict[str, Any] = {"inputs": params}
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(payload, {"__builtins__": allowed_builtins}, locals_env)
            safe_locals = {
                key: _safe_jsonable(value)
                for key, value in locals_env.items()
                if not key.startswith("__")
            }
            out_queue.put({
                "stdout": stdout_capture.getvalue(),
                "locals": safe_locals,
            })
        except Exception as exc:  # pragma: no cover - defensive
            out_queue.put({
                "error": f"{type(exc).__name__}: {exc}",
                "stdout": stdout_capture.getvalue(),
                "traceback": traceback.format_exc(),
            })

    process = multiprocessing.Process(target=_worker, args=(code, inputs, queue), daemon=True)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(1)
        raise TimeoutError("Code execution timed out")

    if queue.empty():
        return {"stdout": "", "locals": {}, "warning": "no output"}

    return queue.get_nowait()


def tool_convert_to_text(session_dirs: Dict[str, str], progress_area=None) -> Dict[str, object]:
    """Convert uploaded inputs into text into examined_txt using shared converters."""

    paths = get_agent_paths(session_dirs)
    uploads = paths["uploads_inputs"]
    txt_dir = paths["examined_txt"]
    os.makedirs(txt_dir, exist_ok=True)

    created: List[str] = []
    # Snapshot current uploads directory content for debugging/traceability
    try:
        uploads_listing = sorted(os.listdir(uploads)) if os.path.isdir(uploads) else []
    except Exception:
        uploads_listing = []
    # A lightweight progress shim
    class _Pg:
        def info(self, msg: str) -> None:
            if progress_area is not None:
                try:
                    progress_area.info(msg)
                except Exception:
                    pass
        write = info
        warning = info
        error = info

    pg = progress_area or _Pg()

    created += process_pdf_folder(uploads, txt_dir, pg, annotate_sources=True)
    created += process_word_ppt_folder(uploads, txt_dir, pg, annotate_sources=True)
    created += process_excel_folder(uploads, txt_dir, pg, annotate_sources=True)
    created += process_textlike_folder(uploads, txt_dir, pg) or []
    process_archives(uploads, txt_dir, pg)
    # Also report any pre-existing txts (from earlier runs)
    try:
        existing_txts = [
            os.path.join(txt_dir, name)
            for name in sorted(os.listdir(txt_dir))
            if os.path.isfile(os.path.join(txt_dir, name)) and name.lower().endswith(".txt")
        ]
    except Exception:
        existing_txts = []

    return {
        "uploads": uploads,
        "uploads_files": uploads_listing,
        "examined_txt": txt_dir,
        "files": created,
        "existing_txts": existing_txts,
    }


__all__ = [
    "prepare_conversation_dirs",
    "get_agent_paths",
    "tool_filesystem",
    "tool_http_fetch",
    "tool_convert_to_text",
    "tool_web_search",
    "tool_python_exec",
]


