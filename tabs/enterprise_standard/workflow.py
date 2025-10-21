"""Enterprise standard workflow surface configuration."""
from __future__ import annotations

from tabs.shared import WorkflowSurface

from .settings import TAB_SLUG

ENTERPRISE_CHUNK_PROMPT_PREFIX = (
    "请作为企业标准符合性检查专家，审阅待检查文件与企业标准是否一致。"
    "以列表形式列出不一致的点，并引用原文证据（简短摘录）、标明出处（提供企业标准文件的文件名）。\n"
    "输出的内容要言简意赅，列出不一致的点即可，不需要列出一致的点，也不需要列出企业标准中缺失的点，最后不需要总结。\n"
    "由于待检查文件较长，我将分成多个部分将其上传给你。以下是待检查文件的一部分。\n"
)

ENTERPRISE_WORKFLOW_SURFACE = WorkflowSurface(
    slug=TAB_SLUG,
    output_subdir="enterprise_standard_check",
    chunk_prompt_prefix=ENTERPRISE_CHUNK_PROMPT_PREFIX,
    standards_dir_key="enterprise_standards",
    examined_dir_key="enterprise_examined",
    warmup_prompt="预热：请简短回复 'gotcha' 即可。",
)

__all__ = ["ENTERPRISE_WORKFLOW_SURFACE", "ENTERPRISE_CHUNK_PROMPT_PREFIX"]
