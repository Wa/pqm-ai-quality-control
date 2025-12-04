"""Parameters workflow surface configuration."""
from __future__ import annotations

from tabs.shared import WorkflowSurface

from .settings import TAB_SLUG

PARAMETERS_CHUNK_PROMPT_PREFIX = (
    "你是一名设计制程参数一致性检查专家。请对照基准文件，审阅待检查文件及其图纸中是否存在与基准不一致的参数。"
    "请聚焦于参数名称、目标值/范围、基准值/范围以及图纸标注的差异。若无不一致，可说明“未发现不一致项”。\n"
    "\n对于每一处不一致，请提供以下信息："
    "1) 参数或特性名称（即条目）；"
    "2) 基准文件名称和工作表名（如果有工作表名信息）；"
    "3) 基准文件中的参数取值/范围；"
    "4) 待检查文件或图纸的文件名称和工作表名（如果有工作表名信息）；"
    "5) 待检查文件或图纸中的参数取值/范围；"
    "6) 问题描述（一句话描述不一致或风险）；"
    "\n要求和注意事项："
    "1) 请将待检查文件中的所有参数进行逐一检查。"
    "2) 参数名称的具体文字表述可能有一些差异，你需要根据意思是判断是否是同一个参数。"
    "3) 如果一个参数仅在基准文件中出现，而在待检查文件中没有出现，则不需要列出。"
    "4) 同样地，如果一个参数仅在待检查文件中出现，而在基准文件中没有出现，则不需要列出。"
    "5) 不需要列出一致的参数。"
    "6) 若未发现参数不一致，可输出“未发现不一致项”，不需要说明理由。\n"
    "输出时保持简洁。以下是待检查文件《{source_file}》内容片段：\n"
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
