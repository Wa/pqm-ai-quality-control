"""Shared helpers reusable across multiple Streamlit tabs."""

from .common import estimate_tokens, report_exception, stream_text
from .file_conversion import (
    cleanup_orphan_txts,
    process_archives,
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)
from .metrics import log_llm_metrics
from .text_processing import (
    annotate_txt_file_inplace,
    insert_source_markers,
    preprocess_txt_directories,
)
from .workflows import WorkflowPaths, WorkflowSurface

__all__ = [
    "WorkflowPaths",
    "WorkflowSurface",
    "estimate_tokens",
    "report_exception",
    "stream_text",
    "process_pdf_folder",
    "process_word_ppt_folder",
    "process_excel_folder",
    "process_textlike_folder",
    "process_archives",
    "cleanup_orphan_txts",
    "preprocess_txt_directories",
    "annotate_txt_file_inplace",
    "insert_source_markers",
    "log_llm_metrics",
]
