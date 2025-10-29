"""Special symbols workflow surface configuration."""
from __future__ import annotations

from tabs.shared import WorkflowSurface

from .settings import TAB_SLUG

SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX = (
    "你是一名特殊特性分类（★、☆、/）检查专家，需根据基准文件核对待检查文件。"
    "请聚焦于待检查文件中特殊特性符号的标注是否存在与基准文件不一致的情况。\n"
    "对于每一处不一致，请提供具体的条目（即特征、特性、过程或产品）、基准文件名和所述条目在基准文件中的位置、"
    "基准文件中的特殊特性分类（★、☆、/）、待检查文件名、和待检查文件中的特殊特性分类（★、☆、/）。\n"
    "\n要求和注意事项："
    "1) 请将待检查文件中的所有条目进行逐一检查。"
    "2) 尽量保留可用于追溯定位的原文线索（如文件名、SHEET名、页码等）于相应字段中。"
    "3) 斜杠，即/，不总是代表特殊特性分类，有时斜杠代表该处无内容，请结合上下文判断。"
    "4) 条目的具体文字表述可能有一些差异，你需要根据意思是判断是否是同一个条目。"
    "5) 如果一个条目仅在基准文件中出现，而在待检查文件中没有出现，则不需要列出，反之亦然。"
    "6) 不需要列出一致的条目。"
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
