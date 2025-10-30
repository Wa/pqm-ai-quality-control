"""Parameters workflow surface configuration."""
from __future__ import annotations

from tabs.shared import WorkflowSurface

from .settings import TAB_SLUG

PARAMETERS_CHUNK_PROMPT_PREFIX = (
    "你是一名设计制程参数一致性检查专家。请对照基准文件，审阅待检查文件及其图纸中是否存在与基准不一致的参数。"
    "请聚焦于参数名称、目标值/范围、基准值/范围以及图纸标注的差异。若无不一致，可说明“未发现不一致项”。\n"
    "输出时保持简洁，逐条列出发现的不一致点，并引用来源文件名以便追溯。以下是待检查文件内容片段：\n"
)

PARAMETERS_WORKFLOW_SURFACE = WorkflowSurface(
    slug=TAB_SLUG,
    output_subdir="parameters_check",
    chunk_prompt_prefix=PARAMETERS_CHUNK_PROMPT_PREFIX,
    standards_dir_key="reference",
    examined_dir_key="target",
    warmup_prompt="预热：请回复'参数检查准备就绪'。",
)

__all__ = ["PARAMETERS_WORKFLOW_SURFACE", "PARAMETERS_CHUNK_PROMPT_PREFIX"]
