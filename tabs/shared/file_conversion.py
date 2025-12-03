"""File conversion helpers shared across document-processing tabs."""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

import requests

from config import CONFIG
from pdf_to_structured_txt import extract_pdf_to_txt, has_vector_text
from .text_processing import annotate_txt_file_inplace


ZIP_EXTS = {".docx", ".pptx", ".xlsx"}
OLE_EXTS = {".doc", ".ppt", ".xls"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt"}

EXPECTED_SIGNATURES = {
    ".docx": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
    ".doc": b"\xD0\xCF\x11\xE0",
    ".ppt": b"\xD0\xCF\x11\xE0",
    ".xls": b"\xD0\xCF\x11\xE0",
    ".pdf": b"%PDF",
}


def detect_esafenet_encryption(path: Path) -> Tuple[bool, str]:
    """
    Return (likely_encrypted, reason) for a single file.

    Reason indicates which structural check failed; downstream callers can
    decide how to label/report it.
    """

    ext = path.suffix.lower()
    expected = EXPECTED_SIGNATURES.get(ext, b"")
    header_size = max(4, len(expected))

    try:
        with path.open("rb") as fh:
            signature = fh.read(header_size)
    except OSError as exc:
        return True, f"unreadable file: {exc}"

    if expected and not signature.startswith(expected):
        return True, f"missing expected header for {ext or 'unknown'}"

    if ext in ZIP_EXTS:
        try:
            with zipfile.ZipFile(path) as zf:
                if "[Content_Types].xml" not in zf.namelist():
                    return True, "Office manifest missing"
        except zipfile.BadZipFile:
            return True, "corrupted Office ZIP container"

    if ext in OLE_EXTS and not signature.startswith(b"\xD0\xCF\x11\xE0"):
        return True, "missing OLE Compound header"

    if ext in PDF_EXTS and not signature.startswith(b"%PDF"):
        return True, "missing PDF header"

    if ext in TEXT_EXTS:
        try:
            signature.decode("utf-8")
        except UnicodeDecodeError:
            return True, "text file contains binary bytes"

    return False, "signature checks passed"


def _esafenet_warning(name: str) -> str:
    return f"《{name}》经亿赛通加密，无法读取，请上传解密版本"


def _is_encrypted(path: str, progress_area, *, warn: bool = True) -> bool:
    encrypted, _reason = detect_esafenet_encryption(Path(path))
    if encrypted and warn:
        progress_area.warning(_esafenet_warning(os.path.basename(path)))
    return encrypted


def list_pdfs(folder: str) -> List[str]:
    """Return absolute paths for all PDF files in a folder (non-recursive)."""

    try:
        return [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(".pdf")
        ]
    except Exception:
        return []


def mineru_parse_pdf(pdf_path: str) -> bytes:
    """Call MinerU API to parse a single PDF and return ZIP bytes on success."""

    api_url = "http://10.31.60.127:8000/file_parse"
    data = {
        "backend": "vlm-sglang-engine",
        "response_format_zip": "true",
        "formula_enable": "true",
        "table_enable": "true",
        "return_images": "false",
        "return_middle_json": "true",
        "return_model_output": "false",
        "return_content_list": "true",
    }
    with open(pdf_path, "rb") as handle:
        files = {"files": (os.path.basename(pdf_path), handle, "application/pdf")}
        resp = requests.post(api_url, data=data, files=files, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"MinerU API error {resp.status_code}: {resp.text[:200]}")
        return resp.content


def zip_to_txts(zip_bytes: bytes, target_txt_path: str) -> bool:
    """Extract first .md file from ZIP bytes and save as plain text (.txt)."""

    bio = io.BytesIO(zip_bytes)
    try:
        with zipfile.ZipFile(bio) as zf:
            md_members = [name for name in zf.namelist() if name.lower().endswith(".md")]
            if not md_members:
                return False
            name = md_members[0]
            with zf.open(name) as md_file:
                content = md_file.read().decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        return False

    with open(target_txt_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return True


def process_pdf_folder(
    input_dir: str,
    output_dir: str,
    progress_area,
    annotate_sources: bool = False,
    *,
    skip_files: Optional[Set[str]] = None,
    warn_if_encrypted: bool = True,
    use_structured_drawings: bool = False,
    structured_word_threshold: int = 15,
):
    """Convert PDFs to text, optionally using structured extraction for drawings."""

    pdf_paths = list_pdfs(input_dir)
    if not pdf_paths:
        return []
    created: List[str] = []
    for pdf_path in pdf_paths:
        orig_name = os.path.basename(pdf_path)
        if skip_files and orig_name in skip_files:
            if warn_if_encrypted:
                progress_area.warning(_esafenet_warning(orig_name))
            continue
        if warn_if_encrypted and _is_encrypted(pdf_path, progress_area):
            continue
        out_txt = os.path.join(output_dir, f"{orig_name}.txt")
        try:
            if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                progress_area.info(f"已存在（跳过）: {os.path.basename(out_txt)}")
                continue
            used_structured = False
            ok = False

            if use_structured_drawings:
                try:
                    if has_vector_text(
                        pdf_path,
                        min_words=max(structured_word_threshold, 1),
                        sample_pages=3,
                    ):
                        progress_area.write(f"结构化解析: {os.path.basename(pdf_path)} …")
                        extract_pdf_to_txt(pdf_path, out_txt)
                        used_structured = True
                        ok = True
                    else:
                        progress_area.info(
                            f"检测为扫描件或文本稀少，改用通用解析: {os.path.basename(pdf_path)}"
                        )
                except Exception as exc:
                    progress_area.warning(
                        f"结构化解析失败，改用通用流程: {exc}"
                    )

            if not ok:
                progress_area.write(f"解析: {os.path.basename(pdf_path)} …")
                zip_bytes = mineru_parse_pdf(pdf_path)
                ok = zip_to_txts(zip_bytes, out_txt)

            if ok:
                if annotate_sources:
                    annotate_txt_file_inplace(out_txt, orig_name)
                created.append(out_txt)
            elif used_structured:
                progress_area.warning(
                    f"结构化解析失败且未找到可用文本: {os.path.basename(pdf_path)}"
                )
            else:
                progress_area.warning(
                    f"未发现可用的 .md 内容，跳过: {os.path.basename(pdf_path)}"
                )
        except Exception as exc:  # pragma: no cover - Streamlit UI feedback
            progress_area.error(f"失败: {os.path.basename(pdf_path)} → {exc}")
    return created


def list_word_ppt(folder: str) -> List[str]:
    """Return absolute paths for .doc, .docx, .ppt, .pptx in a folder (non-recursive)."""

    try:
        return [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and os.path.splitext(name)[1].lower() in {".doc", ".docx", ".ppt", ".pptx"}
        ]
    except Exception:
        return []


def unstructured_partition_to_txt(file_path: str, target_txt_path: str) -> bool:
    """Send a single Word/PPT file to Unstructured API and write plain text (.txt)."""

    api_url = (
        os.getenv("UNSTRUCTURED_API_URL")
        or CONFIG.get("services", {}).get("unstructured_api_url")
        or "http://10.31.60.11:8000/general/v0/general"
    )
    try:
        with open(file_path, "rb") as handle:
            files = {"files": (os.path.basename(file_path), handle)}
            form = {
                "strategy": "auto",
                "ocr_languages": "chi_sim,eng",
                "infer_table_structure": "true",
            }
            resp = requests.post(api_url, files=files, data=form, timeout=300)
            resp.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Unstructured API error: {error}") from error

    try:
        payload = resp.json()
    except ValueError as error:
        raise RuntimeError(f"Unstructured response JSON decode failed: {error}") from error

    blocks = payload if isinstance(payload, list) else payload.get("elements", [])
    if not isinstance(blocks, list):
        raise RuntimeError("Unexpected Unstructured response format")

    lines: List[str] = []
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        if text:
            lines.append(text)
    content = "\n".join(lines)
    with open(target_txt_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return True


def process_word_ppt_folder(
    input_dir: str,
    output_dir: str,
    progress_area,
    annotate_sources: bool = False,
    *,
    skip_files: Optional[Set[str]] = None,
    warn_if_encrypted: bool = True,
):
    paths = list_word_ppt(input_dir)
    if not paths:
        return []
    created: List[str] = []
    for path in paths:
        orig_name = os.path.basename(path)
        if skip_files and orig_name in skip_files:
            if warn_if_encrypted:
                progress_area.warning(_esafenet_warning(orig_name))
            continue
        if warn_if_encrypted and _is_encrypted(path, progress_area):
            continue
        out_txt = os.path.join(output_dir, f"{orig_name}.txt")
        try:
            if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                progress_area.info(f"已存在（跳过）: {os.path.basename(out_txt)}")
                continue
            progress_area.write(f"解析: {orig_name} …")
            if unstructured_partition_to_txt(path, out_txt):
                if annotate_sources:
                    annotate_txt_file_inplace(out_txt, orig_name)
                created.append(out_txt)
        except Exception as exc:  # pragma: no cover - Streamlit UI feedback
            progress_area.error(f"失败: {orig_name} → {exc}")
    return created


def list_excels(folder: str) -> List[str]:
    try:
        return [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and os.path.splitext(name)[1].lower() in {".xls", ".xlsx", ".xlsm"}
        ]
    except Exception:
        return []


def sanitize_sheet_name(name: str) -> str:
    bad = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for ch in bad:
        name = name.replace(ch, "_")
    return "_".join(name.strip().split())[:80] or "Sheet"


def process_excel_folder(
    input_dir: str,
    output_dir: str,
    progress_area,
    annotate_sources: bool = False,
    *,
    skip_files: Optional[Set[str]] = None,
    warn_if_encrypted: bool = True,
):
    paths = list_excels(input_dir)
    if not paths:
        return []
    created: List[str] = []
    try:
        import pandas as pd  # type: ignore
    except ImportError as error:
        progress_area.warning(f"未安装 pandas，无法处理 Excel：{error}")
        return []
    for excel_path in paths:
        orig_name = os.path.basename(excel_path)
        if skip_files and orig_name in skip_files:
            if warn_if_encrypted:
                progress_area.warning(_esafenet_warning(orig_name))
            continue
        if warn_if_encrypted and _is_encrypted(excel_path, progress_area):
            continue
        try:
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                safe_sheet = sanitize_sheet_name(sheet)
                out_txt = os.path.join(output_dir, f"{orig_name}_SHEET_{safe_sheet}.txt")
                if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                    progress_area.info(f"已存在（跳过）: {os.path.basename(out_txt)}")
                    continue
                progress_area.write(f"转换(Excel→CSV): {orig_name} / {sheet} …")
                df = xls.parse(sheet)
                df.to_csv(out_txt, index=False, encoding="utf-8")
                if annotate_sources:
                    annotate_txt_file_inplace(out_txt, f"{orig_name} / {sheet}")
                created.append(out_txt)
        except Exception as exc:  # pragma: no cover - Streamlit UI feedback
            progress_area.error(f"失败: {orig_name} → {exc}")
    return created


def process_textlike_folder(
    input_dir: str,
    output_dir: str,
    progress_area,
    *,
    skip_files: Optional[Set[str]] = None,
    warn_if_encrypted: bool = True,
):
    try:
        if not os.path.isdir(input_dir):
            return []
        exts = {
            ".txt",
            ".md",
            ".csv",
            ".tsv",
            ".json",
            ".yaml",
            ".yml",
            ".log",
            ".ini",
            ".cfg",
            ".rst",
        }
        os.makedirs(output_dir, exist_ok=True)
        written: List[str] = []
        for name in os.listdir(input_dir):
            src_path = os.path.join(input_dir, name)
            if not os.path.isfile(src_path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            if skip_files and name in skip_files:
                if warn_if_encrypted:
                    progress_area.warning(_esafenet_warning(name))
                continue
            if warn_if_encrypted and _is_encrypted(src_path, progress_area):
                continue
            base = os.path.splitext(name)[0]
            dst = os.path.join(output_dir, f"{base}.txt")
            try:
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    progress_area.info(f"已存在（跳过）: {os.path.basename(dst)}")
                    continue
                with open(src_path, "r", encoding="utf-8", errors="ignore") as reader:
                    content = reader.read()
                with open(dst, "w", encoding="utf-8") as writer:
                    writer.write(content)
                written.append(dst)
            except Exception as exc:  # pragma: no cover - Streamlit UI feedback
                progress_area.warning(f"复制失败: {name} → {exc}")
        return written
    except Exception:
        return []


def process_archives(input_dir: str, output_dir: str, progress_area) -> int:
    try:
        if not os.path.isdir(input_dir):
            return 0
        processed = 0
        try:
            import py7zr  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            py7zr = None  # type: ignore
        try:
            import rarfile  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            rarfile = None  # type: ignore

        for name in os.listdir(input_dir):
            src_path = os.path.join(input_dir, name)
            if not os.path.isfile(src_path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in {".zip", ".7z", ".rar"}:
                continue
            tmp_root = tempfile.mkdtemp(prefix="extract_", dir=output_dir)
            ok = False
            try:
                if ext == ".zip":
                    try:
                        with zipfile.ZipFile(src_path) as zf:
                            zf.extractall(tmp_root)
                        ok = True
                    except Exception as exc:
                        progress_area.warning(f"解压失败: {name} → {exc}")
                elif ext == ".7z" and py7zr is not None:
                    try:
                        with py7zr.SevenZipFile(src_path, mode="r") as archive:
                            archive.extractall(path=tmp_root)
                        ok = True
                    except Exception as exc:
                        progress_area.warning(f"解压7z失败: {name} → {exc}")
                elif ext == ".rar" and rarfile is not None:
                    try:
                        archive = rarfile.RarFile(src_path)
                        archive.extractall(tmp_root)
                        archive.close()
                        ok = True
                    except Exception as exc:
                        progress_area.warning(f"解压rar失败: {name} → {exc}")
                else:
                    progress_area.info(f"跳过未支持的压缩包: {name}")

                if ok:
                    for root, _dirs, _files in os.walk(tmp_root):
                        process_pdf_folder(root, output_dir, progress_area, annotate_sources=True)
                        process_word_ppt_folder(root, output_dir, progress_area, annotate_sources=True)
                        process_excel_folder(root, output_dir, progress_area, annotate_sources=True)
                        process_textlike_folder(root, output_dir, progress_area)
                    processed += 1
            finally:
                try:
                    shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception:
                    pass
        return processed
    except Exception:
        return 0


def cleanup_orphan_txts(source_dir: str, output_dir: str, progress_area=None) -> int:
    try:
        if not os.path.isdir(output_dir):
            return 0
        keep_exact = set()
        keep_prefixes: List[str] = []
        try:
            for fname in os.listdir(source_dir or "."):
                src_path = os.path.join(source_dir, fname)
                if not os.path.isfile(src_path):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in {".pdf", ".doc", ".docx", ".ppt", ".pptx"}:
                    keep_exact.add((fname + ".txt").lower())
                elif ext in {".xls", ".xlsx", ".xlsm"}:
                    keep_prefixes.append((fname + "_SHEET_").lower())
        except Exception:
            return 0

        deleted_count = 0
        for name in os.listdir(output_dir):
            out_path = os.path.join(output_dir, name)
            if not os.path.isfile(out_path):
                continue
            name_lower = name.lower()
            if not name_lower.endswith(".txt"):
                continue
            keep = name_lower in keep_exact or any(name_lower.startswith(prefix) for prefix in keep_prefixes)
            if not keep:
                try:
                    os.remove(out_path)
                    deleted_count += 1
                    if progress_area is not None:
                        progress_area.info(f"清理无关文本: {name}")
                except Exception:
                    pass
        return deleted_count
    except Exception:
        return 0


__all__ = [
    "cleanup_orphan_txts",
    "list_excels",
    "list_pdfs",
    "list_word_ppt",
    "mineru_parse_pdf",
    "process_archives",
    "process_excel_folder",
    "process_pdf_folder",
    "process_textlike_folder",
    "process_word_ppt_folder",
    "sanitize_sheet_name",
    "unstructured_partition_to_txt",
    "zip_to_txts",
]
