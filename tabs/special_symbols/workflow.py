"""Special symbols workflow surface configuration."""
from __future__ import annotations

from tabs.shared import WorkflowSurface

from .settings import TAB_SLUG

SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX = (
    "你是一名特殊特性符号检查专家，需根据企业参考资料核对待检文件。"
    "请聚焦于符号的标注是否存在遗漏、错误或与标准不一致的情况。\n"
    "对于每一处问题，请提供简要说明并引用参考资料的出处（文件名及关键位置）。\n"
    "以下是待检文件的一部分：\n"
)

SPECIAL_SYMBOLS_WORKFLOW_SURFACE = WorkflowSurface(
    slug=TAB_SLUG,
    output_subdir="special_symbols_check",
    chunk_prompt_prefix=SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX,
    standards_dir_key="special_reference",
    examined_dir_key="special_examined",
    warmup_prompt="预热：请回复 'ready for symbols'。",
)

__all__ = ["SPECIAL_SYMBOLS_WORKFLOW_SURFACE", "SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX"]
