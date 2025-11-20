"""Backend job runner for文件要素检查."""
from __future__ import annotations

import os
import shutil
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

from config import CONFIG

from .config import DeliverableProfile, ElementRequirement
from .evaluator import EvaluationOrchestrator, parse_deliverable_stub, save_result_payload


def _build_requirement(item: Dict[str, object], index: int) -> ElementRequirement:
    return ElementRequirement(
        key=str(item.get("key") or f"item_{index}"),
        name=str(item.get("name") or f"要素{index + 1}"),
        severity=str(item.get("severity") or "major"),
        description=str(item.get("description") or item.get("guidance") or "—"),
        guidance=str(item.get("guidance") or item.get("description") or "—"),
        keywords=(),
    )


def _build_profile(payload: Dict[str, object]) -> DeliverableProfile:
    requirements_payload = payload.get("requirements") or []
    requirement_objects: List[ElementRequirement] = []
    if isinstance(requirements_payload, Sequence):
        for index, item in enumerate(requirements_payload):
            if isinstance(item, dict):
                requirement_objects.append(_build_requirement(item, index))

    if not requirement_objects:
        raise ValueError("未提供有效的要素要求，无法启动评估。")

    references_field = payload.get("references")
    if isinstance(references_field, Sequence) and not isinstance(references_field, (str, bytes)):
        references = tuple(str(item) for item in references_field)
    else:
        references = ()

    return DeliverableProfile(
        id=str(payload.get("id") or payload.get("profile_id") or "file_elements"),
        stage=str(payload.get("stage") or ""),
        name=str(payload.get("name") or "自定义交付物"),
        description=str(payload.get("description") or ""),
        references=references,
        requirements=tuple(requirement_objects),
    )


def run_file_elements_job(
    session_id: str,
    publish: Callable[[Dict[str, object]], None],
    *,
    profile_payload: Dict[str, object],
    source_paths: Optional[Sequence[str]] = None,
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
) -> None:
    """Execute the文件要素检查 workflow inside the backend worker."""

    publish(
        {
            "status": "running",
            "stage": "initializing",
            "message": "准备AI要素评估配置",
            "progress": 0.0,
        }
    )

    uploads_root = str(CONFIG["directories"]["uploads"])
    generated_root = str(CONFIG["directories"]["generated_files"])
    source_dir = os.path.join(uploads_root, session_id, "elements")
    parsed_dir = os.path.join(generated_root, session_id, "file_elements_check", "parsed_files")
    export_dir = os.path.join(generated_root, session_id, "file_elements_check")
    final_results_dir = os.path.join(generated_root, session_id, "file_elements_check", "final_results")

    for path in (source_dir, parsed_dir, export_dir, final_results_dir):
        os.makedirs(path, exist_ok=True)

    def _reset_uploads_directory() -> None:
        shutil.rmtree(source_dir, ignore_errors=True)
        os.makedirs(source_dir, exist_ok=True)

    try:
        profile = _build_profile(profile_payload)
    except ValueError as error:
        publish(
            {
                "status": "failed",
                "stage": "initializing",
                "message": str(error),
                "error": str(error),
                "progress": 0.0,
            }
        )
        _reset_uploads_directory()
        return

    try:
        normalized_paths: List[str] = []
        if source_paths:
            for item in source_paths:
                if not item:
                    continue
                normalized = os.path.normpath(str(item))
                if normalized not in normalized_paths:
                    normalized_paths.append(normalized)

        publish(
            {
                "status": "running",
                "stage": "conversion",
                "message": "正在解析上传文件",
                "progress": 2.0,
            }
        )

        text, source_file, warnings = parse_deliverable_stub(
            profile,
            source_dir,
            parsed_dir,
            source_paths=normalized_paths,
        )

        progress_value = 10.0
        publish(
            {
                "status": "running",
                "stage": "conversion",
                "message": "已完成文本解析，准备调用大模型",
                "progress": progress_value,
            }
        )

        orchestrator = EvaluationOrchestrator(profile)
        result_holder: Dict[str, object] = {}
        error_holder: Dict[str, BaseException] = {}

        def _run_orchestrator() -> None:
            try:
                result_holder["result"] = orchestrator.evaluate(
                    text,
                    source_file=source_file,
                    warnings=warnings,
                )
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc

        worker = threading.Thread(target=_run_orchestrator, daemon=True)
        worker.start()
        pseudo_progress = progress_value

        while worker.is_alive():
            delay = 2.0 if pseudo_progress < 80.0 else 20.0
            time.sleep(delay)
            if not worker.is_alive():
                break
            pseudo_progress = min(pseudo_progress + 1.0, 99.0)
            publish(
                {
                    "status": "running",
                    "stage": "evaluating",
                    "message": "大模型正在分析交付物…",
                    "progress": pseudo_progress,
                }
            )

        worker.join()

        if "error" in error_holder:
            publish(
                {
                    "status": "failed",
                    "stage": "completed",
                    "message": f"评估失败：{error_holder['error']}",
                    "error": str(error_holder["error"]),
                    "progress": pseudo_progress,
                }
            )
            return

        result = result_holder.get("result")
        if result is None:
            publish(
                {
                    "status": "failed",
                    "stage": "completed",
                    "message": "评估未返回结果，请稍后重试。",
                    "error": "empty_result",
                    "progress": pseudo_progress,
                }
            )
            return

        saved_path = save_result_payload(result, export_dir)
        tabular_exports = result.export_tabular(final_results_dir, base_filename=source_file)
        result_files: List[str] = []
        if saved_path:
            result_files.append(saved_path)
        result_files.extend(path for path in tabular_exports.values() if path)
        publish(
            {
                "status": "succeeded",
                "stage": "completed",
                "message": "评估完成，可下载结果。",
                "progress": 100.0,
                "result_files": result_files,
                "metadata": {
                    "stage": profile.stage,
                    "deliverable": profile.name,
                    "source_file": source_file,
                },
            }
        )
    finally:
        _reset_uploads_directory()


__all__ = ["run_file_elements_job"]
