"""Metrics logging utilities shared across Streamlit workflows."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict

import streamlit as st

from .common import report_exception


def _get_metrics_path(base_out_dir: str, session_id: str) -> str:
    metrics_dir = os.path.join(base_out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    key = f"metrics_file_{session_id}"
    fname = st.session_state.get(key)
    if not fname:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"llm_calls_{ts}.csv"
        st.session_state[key] = fname
    return os.path.join(metrics_dir, fname)


def log_llm_metrics(base_out_dir: str, session_id: str, row: Dict) -> None:
    try:
        path = _get_metrics_path(base_out_dir, session_id)
        exists = os.path.exists(path)
        headers = [
            "ts",
            "engine",
            "model",
            "session_id",
            "file",
            "part",
            "phase",
            "prompt_chars",
            "prompt_tokens",
            "output_chars",
            "output_tokens",
            "duration_ms",
            "success",
            "error",
        ]
        with open(path, "a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(headers)
            writer.writerow(
                [
                    row.get("ts"),
                    row.get("engine"),
                    row.get("model"),
                    row.get("session_id"),
                    row.get("file"),
                    row.get("part"),
                    row.get("phase"),
                    row.get("prompt_chars"),
                    row.get("prompt_tokens"),
                    row.get("output_chars"),
                    row.get("output_tokens"),
                    row.get("duration_ms"),
                    row.get("success"),
                    row.get("error"),
                ]
            )
    except Exception as error:  # pragma: no cover - defensive UI feedback
        report_exception("写入LLM指标失败", error, level="warning")


__all__ = ["log_llm_metrics"]
