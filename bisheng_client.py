# ascii-touch: test write path
import io
import json
import os
import re
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import requests


class _Http:
    session: requests.Session = requests.Session()
    try:
        from requests.adapters import HTTPAdapter  # type: ignore
        from urllib3.util.retry import Retry  # type: ignore

        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[502, 503, 504]),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    except Exception:
        pass


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    heads = {"Content-Type": "application/json"}
    if api_key:
        heads["Authorization"] = f"Bearer {api_key}"
    return heads


def _headers_upload(api_key: Optional[str]) -> Dict[str, str]:
    heads: Dict[str, str] = {}
    if api_key:
        heads["Authorization"] = f"Bearer {api_key}"
    return heads


# ---------------------------
# Knowledge Base (Filelib) APIs
# ---------------------------

def get_knowledge_list(base_url: str, api_key: Optional[str], name: Optional[str] = None, timeout_s: int = 15) -> List[dict]:
    """Return list of knowledge bases. Optional filter by name."""
    url = base_url.rstrip('/') + "/api/v2/filelib/"
    params: Dict[str, object] = {}
    if name:
        params["name"] = name
    try:
        resp = _Http.session.get(url, headers=_headers(api_key), params=params, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        items = ((data.get("data") or {}).get("data") or [])
        return items if isinstance(items, list) else []
    except Exception:
        return []


def find_knowledge_id_by_name(base_url: str, api_key: Optional[str], kb_name: str, timeout_s: int = 15) -> Optional[int]:
    """Find knowledge_id by exact or case-insensitive match."""
    items = get_knowledge_list(base_url, api_key, name=kb_name, timeout_s=timeout_s)
    for it in items:
        if str(it.get("name")) == kb_name:
            try:
                return int(it.get("id"))
            except Exception:
                pass
    for it in items:
        if str(it.get("name", "")).lower() == kb_name.lower():
            try:
                return int(it.get("id"))
            except Exception:
                pass
    items = get_knowledge_list(base_url, api_key, timeout_s=timeout_s)
    for it in items:
        if str(it.get("name", "")).lower() == kb_name.lower():
            try:
                return int(it.get("id"))
            except Exception:
                pass
    return None


def create_knowledge(base_url: str, api_key: Optional[str], name: str, model: str = "11", description: str = "", timeout_s: int = 20) -> Optional[int]:
    """Create a knowledge base. Returns knowledge_id on success."""
    url = base_url.rstrip('/') + "/api/v2/filelib/"
    payload = {"name": name, "model": model}
    if description:
        payload["description"] = description
    try:
        resp = _Http.session.post(url, headers=_headers(api_key), data=json.dumps(payload), timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        kid = (data.get("data") or {}).get("id")
        return int(kid) if kid is not None else None
    except Exception:
        return None


def kb_get_filelist(base_url: str, api_key: Optional[str], knowledge_id: int, timeout_s: int = 30) -> List[dict]:
    url = base_url.rstrip('/') + "/api/v2/filelib/file/list"
    params = {"knowledge_id": knowledge_id, "page_size": 2000, "page_num": 1}
    try:
        resp = _Http.session.get(url, headers=_headers(api_key), params=params, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        items = ((data.get("data") or {}).get("data") or [])
        return items if isinstance(items, list) else []
    except Exception:
        return []


def kb_delete_file(base_url: str, api_key: Optional[str], file_id: int, timeout_s: int = 30) -> bool:
    url = base_url.rstrip('/') + f"/api/v2/filelib/file/{file_id}"
    try:
        resp = _Http.session.delete(url, headers=_headers(api_key), timeout=timeout_s)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def kb_clear_files(base_url: str, api_key: Optional[str], knowledge_id: int, timeout_s: int = 60) -> bool:
    url = base_url.rstrip('/') + f"/api/v2/filelib/clear/{knowledge_id}"
    try:
        resp = _Http.session.delete(url, headers=_headers(api_key), timeout=timeout_s)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def kb_upload_file(
    base_url: str,
    api_key: Optional[str],
    knowledge_id: int,
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    separators: Optional[List[str]] = None,
    separator_rule: Optional[List[str]] = None,
    timeout_s: int = 120,
) -> Optional[dict]:
    if not os.path.isfile(file_path):
        return None
    url = base_url.rstrip('/') + f"/api/v2/filelib/file/{knowledge_id}"
    if separators is None:
        separators = ["\n\n", "\n"]
    if separator_rule is None:
        separator_rule = ["after", "after"]

    data_list: List[Tuple[str, str]] = [("chunk_size", str(chunk_size)), ("chunk_overlap", str(chunk_overlap))]
    for s in separators:
        data_list.append(("separator", s))
    for r in separator_rule:
        data_list.append(("separator_rule", r))

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            resp = _Http.session.post(url, headers=_headers_upload(api_key), data=data_list, files=files, timeout=timeout_s)
            resp.raise_for_status()
            if resp.headers.get("Content-Type", "").lower().startswith("application/json"):
                return resp.json().get("data") or {}
            return {}
    except Exception:
        return None


def kb_sync_folder(
    base_url: str,
    api_key: Optional[str],
    knowledge_id: int,
    folder_path: str,
    clear_first: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    separators: Optional[List[str]] = None,
    separator_rule: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    uploaded: List[str] = []
    deleted: List[str] = []
    skipped: List[str] = []

    if not folder_path or not os.path.isdir(folder_path):
        return {"uploaded": uploaded, "deleted": deleted, "skipped": skipped}

    local_names: List[str] = []
    for name in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path):
            local_names.append(name)

    existing_items = kb_get_filelist(base_url, api_key, knowledge_id)
    existing_by_name = {str(it.get("file_name")): it for it in existing_items}

    if clear_first:
        kb_clear_files(base_url, api_key, knowledge_id)
        existing_by_name = {}

    local_set = set(local_names)
    for fname, item in list(existing_by_name.items()):
        if fname not in local_set:
            fid = item.get("id")
            try:
                if fid is not None and kb_delete_file(base_url, api_key, int(fid)):
                    deleted.append(fname)
            except Exception:
                pass

    for name in local_names:
        if name in existing_by_name:
            skipped.append(name)
            continue
        path = os.path.join(folder_path, name)
        info = kb_upload_file(
            base_url=base_url,
            api_key=api_key,
            knowledge_id=knowledge_id,
            file_path=path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            separator_rule=separator_rule,
        )
        if info is not None:
            uploaded.append(name)
        else:
            skipped.append(name)

    return {"uploaded": uploaded, "deleted": deleted, "skipped": skipped}

def _extract_text_from_json(obj) -> str:
    if not obj:
        return ""

    def extract_from_value(value):
        if isinstance(value, str):
            return value
        return None

    for key in ["output", "answer", "text", "content", "result", "message"]:
        val = obj.get(key)
        if isinstance(val, (str, list, dict)):
            out = extract_from_value(val)
            if out:
                return out

    for key in ["data", "output_schema", "response", "choices", "outputs"]:
        val = obj.get(key)
        if isinstance(val, (str, list, dict)):
            out = extract_from_value(val)
            if out:
                return out

    return json.dumps(obj, ensure_ascii=False)


def _coerce_to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(_extract_text_from_json(item) if isinstance(item, (list, dict)) else str(item))
        return "".join(parts)
    if isinstance(value, dict):
        return _extract_text_from_json(value)
    return str(value)


def strip_think_sections(text: str) -> str:
    start_tag = "<think>"
    end_tag = "</think>"
    result: List[str] = []
    i = 0
    while i < len(text):
        start_idx = text.find(start_tag, i)
        if start_idx == -1:
            result.append(text[i:])
            break
        result.append(text[i:start_idx])
        end_idx = text.find(end_tag, start_idx + len(start_tag))
        if end_idx == -1:
            break
        i = end_idx + len(end_tag)
    return "".join(result)


def upload_standard_files(base_url: str, api_key: Optional[str], folder_path: str) -> List[str]:
    if not folder_path or not os.path.isdir(folder_path):
        return []
    results: List[str] = []
    upload_url = base_url.rstrip('/') + "/api/v1/knowledge/upload"
    for name in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f)}
                resp = _Http.session.post(upload_url, headers=_headers_upload(api_key), files=files, timeout=60)
                resp.raise_for_status()
                if resp.headers.get("Content-Type", "").lower().startswith("application/json"):
                    data = resp.json()
                else:
                    data = {}
                file_path = ((data.get("data") or {}).get("file_path")
                             or (data.get("data") or {}).get("path")
                             or None)
                if file_path:
                    results.append(file_path)
        except Exception:
            continue
    return results


def call_workflow_invoke(
    base_url: str,
    invoke_path: str,
    workflow_id: str,
    user_question: str,
    api_key: Optional[str],
    timeout_s: int,
    standard_file_urls: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Generator[Tuple[str, Optional[str]], None, None]:
    url = base_url.rstrip("/") + invoke_path
    payload: Dict[str, object] = {
        "workflow_id": workflow_id,
        "inputs": {
            "user_question": user_question,
            "user_input": user_question,
        },
    }
    if session_id:
        payload["session_id"] = session_id

    def make_iterator(r: requests.Response) -> Iterable[dict]:
        ctype = (r.headers.get("Content-Type") or "").lower()

        def iter_sse() -> Iterable[dict]:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                text = line.strip()
                if text.lower() in ("null", "[done]"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                try:
                    obj = json.loads(text)
                    yield obj
                except Exception:
                    continue

        def iter_concat() -> Iterable[dict]:
            partial = ""
            depth = 0
            in_str = False
            esc = False
            start = -1
            for chunk in r.iter_content(decode_unicode=True, chunk_size=1):
                if not chunk:
                    continue
                partial += chunk
                i = 0
                while i < len(partial):
                    c = partial[i]
                    if start == -1:
                        if c == '{':
                            start = i
                            depth = 1
                            in_str = False
                            esc = False
                            i += 1
                            continue
                    else:
                        if in_str:
                            if esc:
                                esc = False
                            elif c == '\\':
                                esc = True
                            elif c == '"':
                                in_str = False
                        else:
                            if c == '"':
                                in_str = True
                            elif c == '{':
                                depth += 1
                            elif c == '}':
                                depth -= 1
                                if depth == 0 and start != -1:
                                    obj_str = partial[start:i+1]
                                    try:
                                        yield json.loads(obj_str)
                                    except Exception:
                                        pass
                                    partial = partial[i+1:]
                                    i = -1
                                    start = -1
                    i += 1
            return

        return iter_sse() if "text/event-stream" in ctype else iter_concat()

    def stream_once(current_timeout: int) -> Generator[Tuple[str, Optional[str]], None, None]:
        resp = _Http.session.post(url, headers=_headers(api_key), data=json.dumps(payload), timeout=current_timeout, stream=True)
        resp.raise_for_status()
        buf: List[str] = []
        new_session_id: Optional[str] = None
        iterator = make_iterator(resp)
        while True:
            saw_input = False
            for obj in iterator:
                if not new_session_id:
                    new_session_id = obj.get("session_id") or (obj.get("data") or {}).get("session_id")
                container = obj.get("data") if isinstance(obj.get("data"), dict) else obj
                event_type = container.get("event")
                if event_type in ("input", "output_with_input_msg", "output_with_choose_msg"):
                    input_schema = container.get("input_schema") or {}
                    text_keys: List[str] = []
                    file_keys: List[str] = []
                    for item in (input_schema.get("value") or []):
                        key = item.get("key")
                        typ = (item.get("type") or "").lower()
                        if not key:
                            continue
                        if typ in ("text", "textarea"):
                            text_keys.append(key)
                        elif typ in ("dialog_file", "file"):
                            file_keys.append(key)
                    if not file_keys and any((it.get("key") == "dialog_files_content") for it in (input_schema.get("value") or [])):
                        file_keys.append("dialog_files_content")
                    input_map: Dict[str, object] = {}
                    for k in text_keys:
                        input_map[k] = user_question
                    if file_keys:
                        input_map.update({k: (standard_file_urls or []) for k in file_keys})
                    cont_payload = {
                        "workflow_id": workflow_id,
                        "session_id": new_session_id or session_id,
                        "message_id": container.get("message_id"),
                        "input": {(container.get("node_id") or ""): input_map},
                    }
                    resp_cont = _Http.session.post(url, headers=_headers(api_key), data=json.dumps(cont_payload), timeout=current_timeout, stream=True)
                    resp_cont.raise_for_status()
                    iterator = make_iterator(resp_cont)
                    saw_input = True
                    break
                delta = ""
                if isinstance(container.get("output_schema"), dict):
                    delta = container["output_schema"].get("message") or container["output_schema"].get("text") or ""
                if not delta:
                    delta = container.get("message") or container.get("text") or ""
                delta = _coerce_to_text(delta)
                if delta:
                    cur = "".join(buf)
                    if delta.startswith(cur):
                        buf[:] = [delta]          # 覆盖为服务端累积文本
                    else:
                        buf.append(delta)          # 仅在返回增量时追加
                    yield ("".join(buf), new_session_id)
            if not saw_input:
                break
        yield ("".join(buf), new_session_id)

    last_exc: Optional[Exception] = None
    attempts = [int(timeout_s), max(int(timeout_s) * 2, int(timeout_s) + 90)]
    for t in attempts:
        try:
            for out in stream_once(t):
                yield out
            return
        except (requests.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_exc = e
            continue
        except requests.HTTPError:
            raise
        except Exception as e:
            last_exc = e
            continue

    try:
        ping_payload = {
            "workflow_id": (workflow_id or "").strip() or "test",
            "inputs": {"user_question": "ping"},
        }
        _Http.session.post(url, headers=_headers(api_key), data=json.dumps(ping_payload), timeout=10)
        raise requests.Timeout("Invoke timed out twice; server reachable but not streaming")
    except Exception as ping_exc:  # pragma: no cover - best-effort diagnostics
        raise requests.Timeout(f"Invoke timed out twice; connectivity check failed: {ping_exc}") from last_exc


def split_to_chunks(text: str, max_units: int) -> List[str]:
    def measure_units(s: str) -> int:
        cjk = len(re.findall(r"[\u4E00-\u9FFF]", s))
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", s))
        return cjk + latin_words

    parts = re.split(r"(\n\s*\n+)", text)
    chunks: List[str] = []
    buf: List[str] = []
    count = 0
    for seg in parts:
        seg_units = measure_units(seg)
        if count + seg_units <= max_units or not buf:
            buf.append(seg)
            count += seg_units
        else:
            chunks.append("".join(buf).strip())
            buf = [seg]
            count = seg_units
    if buf:
        chunks.append("".join(buf).strip())

    result: List[str] = []
    for c in chunks:
        if measure_units(c) <= max_units:
            if c:
                result.append(c)
            continue
        sentences = re.split(r"(?<=\.|\?|!|。|？|！)", c)
        acc: List[str] = []
        cnt = 0
        for s in sentences:
            su = measure_units(s)
            if cnt + su > max_units and acc:
                result.append("".join(acc).strip())
                acc = [s]
                cnt = su
            else:
                acc.append(s)
                cnt += su
        if acc:
            result.append("".join(acc).strip())

    final: List[str] = []
    for c in result:
        units = measure_units(c)
        if units <= max_units:
            final.append(c)
            continue
        step = max_units
        for i in range(0, len(c), step):
            final.append(c[i:i + step])
    return [c for c in final if c.strip()]


def aggregate_enterprise_checks(out_root: str) -> Optional[str]:
    initial_dir = os.path.join(out_root, "initial_results")
    if not os.path.isdir(initial_dir):
        return None
    file_paths = [os.path.join(initial_dir, f) for f in os.listdir(initial_dir) if f.lower().endswith(".txt")]
    if not file_paths:
        return None
    file_paths.sort()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = os.path.join(out_root, "final_results")
    os.makedirs(final_dir, exist_ok=True)
    out_path = os.path.join(final_dir, f"企业标准检查汇总_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as outf:
        first = True
        for p in file_paths:
            try:
                with open(p, "r", encoding="utf-8") as inf:
                    raw = inf.read()
            except Exception:
                continue
            clean = strip_think_sections(raw)
            base = os.path.basename(p)
            name_no_ext = os.path.splitext(base)[0]
            display = name_no_ext[9:] if name_no_ext.startswith("response_") else name_no_ext
            if not first:
                outf.write("\n\n")
            first = False
            outf.write(f"下⾯是 {display} 与企业标准匹配的结果：\n")
            outf.write(clean)
    return out_path


def stop_workflow(base_url: str, stop_path: str, session_id: str, api_key: Optional[str], timeout_s: int = 10) -> Dict:
    url = base_url.rstrip("/") + stop_path
    payload = {"session_id": session_id}
    resp = _Http.session.post(url, headers=_headers(api_key), data=json.dumps(payload), timeout=timeout_s)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


