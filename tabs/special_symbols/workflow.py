"""Special symbols workflow surface configuration."""
from __future__ import annotations

from tabs.shared import WorkflowSurface

from .settings import TAB_SLUG

SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX = (
    "你是一名特殊特性符号（/、 ★和☆）检查专家，需根据企业基准文件核对待检查文件。"
    "请聚焦于待检查文件中特殊特性符号的标注是否存在与基准文件不一致的情况。\n"
    "对于每一处问题，请提供简要说明并引用基准文件的出处（文件名及关键位置）。\n"
    "由于每次只发给你待检文件的一部分，有可能该部分并没有和特殊特性符号相关的内容，此时只需返回“无相关发现”即可\n"
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
