"""Standalone runner that mirrors the “交付物要素自动评估” step from the one-click tab,
but skips upload/parse/classify and reuses existing parsed text files.
It submits backend file-elements jobs for 5 parsed files and shows the same progress math
as the original tab (overall_progress_sum/overall_progress_total)."""

import os
import time
import threading
import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from tabs.file_elements import PHASE_TO_DELIVERABLES, DeliverableProfile
from tabs.tab_file_elements_check import _profile_to_payload
from tabs.file_completeness import STAGE_REQUIREMENTS, STAGE_SLUG_MAP

# -----------------------------
# Configuration
# -----------------------------
PARSED_FILES_DIR = r"C:\Users\zzk_j\Documents\pqm-ai-quality-control\generated_files\massive.test\APQP_one_click_check\parsed_files"
SESSION_ID = "massive.test"
GENERATED_ROOT = str(CONFIG["directories"]["generated_files"])
ELEMENTS_INITIAL_RESULTS_DIR = os.path.join(GENERATED_ROOT, SESSION_ID, "APQP_one_click_check", "initial_results_element")
RESULT_ROOT_DIR = os.path.join(GENERATED_ROOT, SESSION_ID, "APQP_one_click_check")
os.makedirs(ELEMENTS_INITIAL_RESULTS_DIR, exist_ok=True)
os.makedirs(RESULT_ROOT_DIR, exist_ok=True)

MAX_JOBS = 5

# -----------------------------
# Helpers mirroring one-click tab
# -----------------------------
# Reverse mapping: slug -> stage name
SLUG_TO_STAGE_NAME = {slug: name for name, slug in STAGE_SLUG_MAP.items()}

def _slug_to_stage_name(slug: str) -> str | None:
    """Convert directory slug (e.g., 'Stage_A') to stage name (e.g., 'A样阶段')."""
    return SLUG_TO_STAGE_NAME.get(slug)

def _resolve_stage_profile(stage_name: str) -> tuple[DeliverableProfile | None, str]:
    """Select a deliverable profile for the stage (mirrors tab logic)."""
    stage_profiles = PHASE_TO_DELIVERABLES.get(stage_name, ())
    if not stage_profiles:
        return None, ""

    requirement_candidates = list(STAGE_REQUIREMENTS.get(stage_name, ()))
    profile_pool = {profile.name: profile for profile in stage_profiles}
    selected_profile: DeliverableProfile | None = None
    display_name = ""

    # Use DELIVERABLE_PROFILE_ALIASES from original tab
    DELIVERABLE_PROFILE_ALIASES = {
        "初始DFMEA": "DFMEA",
        "更新DFMEA": "DFMEA",
        "初始过程流程图": "过程流程图",
        "更新过程流程图": "过程流程图",
        "初版CP": "控制计划",
        "更新CP": "控制计划",
    }

    for candidate in requirement_candidates:
        alias = DELIVERABLE_PROFILE_ALIASES.get(candidate, candidate)
        if alias in profile_pool:
            selected_profile = profile_pool[alias]
            display_name = candidate
            break

    if selected_profile is None:
        selected_profile = stage_profiles[0]
        display_name = selected_profile.name

    return selected_profile, display_name


def _collect_parsed_files(max_files: int = MAX_JOBS) -> list[dict]:
    """Collect up to max_files parsed txt files with stage inferred from directory."""
    collected: list[dict] = []
    if not os.path.isdir(PARSED_FILES_DIR):
        return collected
    for root, _, files in os.walk(PARSED_FILES_DIR):
        stage_name = os.path.basename(root) or "Unknown"
        for name in sorted(files):
            if not name.lower().endswith(".txt"):
                continue
            collected.append(
                {
                    "stage": stage_name,
                    "label": name,
                    "path": os.path.join(root, name),
                }
            )
            if len(collected) >= max_files:
                return collected
    return collected


# -----------------------------
# Session state init
# -----------------------------
if "parsed_entries" not in st.session_state:
    st.session_state.parsed_entries = _collect_parsed_files()

entries = st.session_state.parsed_entries
if len(entries) < MAX_JOBS:
    st.warning(f"只找到 {len(entries)} 个解析文件，需要 {MAX_JOBS} 个文件。请确认目录：{PARSED_FILES_DIR}")
    st.stop()

if "job_progresses" not in st.session_state:
    st.session_state.job_progresses = [0.0] * MAX_JOBS
if "job_statuses" not in st.session_state:
    st.session_state.job_statuses = ["queued"] * MAX_JOBS
if "job_threads" not in st.session_state:
    st.session_state.job_threads = [None] * MAX_JOBS
if "job_ids" not in st.session_state:
    st.session_state.job_ids = [None] * MAX_JOBS
if "profile_info" not in st.session_state:
    st.session_state.profile_info = {}

# -----------------------------
# UI header
# -----------------------------
st.title("Debug: 交付物要素自动评估 Progress Bar")
st.subheader("🧩 交付物要素自动评估（镜像 one-click 元素步骤）")
st.caption(f"使用解析目录：`{PARSED_FILES_DIR}` · 会提交 {MAX_JOBS} 个后台文件要素任务（turbo 模式）")

# -----------------------------
# Backend setup
# -----------------------------
backend_ready = is_backend_available()
backend_client = get_backend_client() if backend_ready else None
if not backend_ready or backend_client is None:
    st.error("后台服务不可用，无法提交要素评估任务。")
    st.stop()

# -----------------------------
# Detect existing running job (session level)
# -----------------------------
active_session_job: dict | None = None
existing_jobs = backend_client.list_file_elements_jobs(SESSION_ID)
if isinstance(existing_jobs, list) and existing_jobs:
    for rec in existing_jobs:
        if not isinstance(rec, dict):
            continue
        if rec.get("status") in {"queued", "running"}:
            active_session_job = rec
            break

# -----------------------------
# Start jobs (mirror payload) but obey single-running-per-session backend rule
# -----------------------------
def _attempt_start_job(idx: int) -> None:
    entry = entries[idx]
    stage_slug = entry["stage"]  # e.g., "Stage_A"
    stage_name = _slug_to_stage_name(stage_slug)  # e.g., "A样阶段"

    if not stage_name:
        st.session_state.job_statuses[idx] = "failed"
        st.warning(f"{stage_slug} 无法映射到阶段名称，跳过 {entry['label']}")
        return

    profile, profile_label = _resolve_stage_profile(stage_name)
    if profile is None:
        st.session_state.job_statuses[idx] = "failed"
        st.warning(f"{stage_name} 未配置要素模板，跳过 {entry['label']}")
        return

    st.session_state.profile_info[idx] = {
        "stage": stage_name,
        "profile": profile_label or profile.name,
    }

    payload = {
        "session_id": SESSION_ID,
        "profile": _profile_to_payload(profile),
        "source_paths": [entry["path"]],
        "turbo_mode": True,
        "initial_results_dir": ELEMENTS_INITIAL_RESULTS_DIR,
        "result_root_dir": RESULT_ROOT_DIR,
    }

    resp = backend_client.start_file_elements_job(payload)
    if isinstance(resp, dict) and resp.get("job_id"):
        st.session_state.job_ids[idx] = resp.get("job_id")
        st.session_state.job_statuses[idx] = "running"
        st.session_state.job_progresses[idx] = 0.0
    else:
        # If backend says there's an existing running job, leave as queued to retry after it finishes
        detail = ""
        if isinstance(resp, dict):
            detail = str(resp.get("detail") or resp.get("message") or "")
        if "已有文件要素检查任务正在运行" in detail:
            st.session_state.job_statuses[idx] = "queued"
        else:
            st.session_state.job_statuses[idx] = "failed"
        st.warning(f"提交任务失败：{resp}")


# Only start a new job if there's no active session-level running/queued job
if not active_session_job:
    for idx in range(MAX_JOBS):
        if st.session_state.job_statuses[idx] == "queued":
            _attempt_start_job(idx)
            break  # start one at a time

# -----------------------------
# Poll statuses & display progress (mirrors tab logic)
# -----------------------------
overall_progress_placeholder = st.container()
overall_progress_total = 0
overall_progress_sum = 0.0

for idx, entry in enumerate(entries[:MAX_JOBS]):
    overall_progress_total += 1
    label = entry["label"]
    stage_slug = entry["stage"]
    stage_name = _slug_to_stage_name(stage_slug) or stage_slug
    
    # Show profile info if available
    profile_info = st.session_state.get("profile_info", {}).get(idx)
    if profile_info:
        st.markdown(f"##### 文件：{label}")
        st.caption(f"阶段: {profile_info['stage']} · 使用交付物: {profile_info['profile']}")
    else:
        st.markdown(f"##### 文件：{label} (阶段: {stage_name})")

    job_id = st.session_state.job_ids[idx]
    status = st.session_state.job_statuses[idx]
    progress_val = st.session_state.job_progresses[idx]
    job_status = None

    if job_id and status in {"running", "queued"}:
        resp = backend_client.get_file_elements_job(job_id)
        if isinstance(resp, dict) and resp.get("job_id"):
            job_status = resp
            status = str(resp.get("status")) or status
            progress_val = float(resp.get("progress") or 0.0)
            st.session_state.job_statuses[idx] = status
            st.session_state.job_progresses[idx] = progress_val
        elif isinstance(resp, dict) and resp.get("detail") == "未找到任务":
            status = "failed"
            st.session_state.job_statuses[idx] = status

    elements_job_status = job_status or {
        "status": status,
        "progress": progress_val,
        "message": "正在处理..." if status in {"queued", "running"} else ("已完成" if status == "succeeded" else "失败"),
    }

    progress_value = float(elements_job_status.get("progress") or 0.0)
    progress_ratio = progress_value / 100.0 if progress_value > 1.0 else progress_value
    progress_ratio = max(0.0, min(progress_ratio, 1.0))
    progress_pct = int(round(progress_ratio * 100))

    bar_col, pct_col = st.columns([9, 1])
    with bar_col:
        st.progress(progress_ratio)
    with pct_col:
        st.markdown(f"**{progress_pct}%**")

    status_value = str(elements_job_status.get("status")) if elements_job_status else ""
    if status_value in {"succeeded", "failed"}:
        overall_progress_sum += 1.0
    elif status_value in {"queued", "running"}:
        overall_progress_sum += progress_ratio

# -----------------------------
# Overall progress (same math as tab)
# -----------------------------
with overall_progress_placeholder:
    st.markdown("---")
    st.markdown("### Overall Progress")
    if overall_progress_total > 0:
        overall_ratio = overall_progress_sum / overall_progress_total
        overall_ratio = max(0.0, min(overall_ratio, 1.0))
        bar_col, pct_col = st.columns([9, 1])
        with bar_col:
            st.progress(overall_ratio)
        with pct_col:
            st.markdown(f"**{int(round(overall_ratio * 100))}%**")
        st.caption(f"Debug: overall_progress_sum = {overall_progress_sum:.2f}, overall_progress_total = {overall_progress_total}")
        st.caption(f"Debug: overall_ratio = {overall_ratio:.4f} = {overall_progress_sum:.2f} / {overall_progress_total}")
    else:
        st.caption("暂无要素评估任务进度。")

# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset All Jobs"):
        st.session_state.job_progresses = [0.0] * MAX_JOBS
        st.session_state.job_statuses = ["queued"] * MAX_JOBS
        st.session_state.job_threads = [None] * MAX_JOBS
        st.session_state.job_ids = [None] * MAX_JOBS
        st.rerun()

with col2:
    st.caption(f"解析文件目录: `{PARSED_FILES_DIR}`")

# Auto-refresh while jobs active
if any(status in {"queued", "running"} for status in st.session_state.job_statuses):
    time.sleep(2)
    st.rerun()

