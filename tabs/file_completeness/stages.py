"""Stage definitions and requirements for the file completeness workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class StageDefinition:
    """Static metadata describing a completeness check stage."""

    name: str
    slug: str
    requirements: Tuple[str, ...]


STAGES: Tuple[StageDefinition, ...] = (
    StageDefinition(
        name="立项阶段",
        slug="Stage_Initial",
        requirements=(
            "项目立项报告",
            "项目可行性分析报告",
            "项目风险评估报告",
            "项目计划书",
            "项目团队组建方案",
            "项目预算方案",
            "项目时间计划",
            "项目质量目标",
            "项目成本目标",
            "项目交付物清单",
        ),
    ),
    StageDefinition(
        name="A样阶段",
        slug="Stage_A",
        requirements=(
            "电芯规格书",
            "尺寸链公差计算书",
            "初始DFMEA",
            "初始特殊特性清单",
            "三新清单",
            "制程标准",
            "开模清单",
            "3D数模",
            "2D图纸",
            "BOM清单",
            "仿真报告",
            "测试大纲",
            "专利挖掘清单",
            "初版PFMEA",
            "产线规划方案",
            "过程设计初始方案",
            "产品可制造性分析及风险应对报告",
            "初始过程流程图",
            "初始过程特殊特性",
            "初版CP",
            "初版SOP",
            "工艺验证计划",
            "样品包装方案",
        ),
    ),
    StageDefinition(
        name="B样阶段",
        slug="Stage_B",
        requirements=(
            "设计变更履历表",
            "更新电芯规格书",
            "更新DFMEA",
            "更新特殊特性清单",
            "制程标准",
            "更新3D数模",
            "更新2D图纸",
            "尺寸链公差计算书",
            "更新BOM清单",
            "更新开模清单",
            "更新三新清单",
            "仿真报告",
            "DV测试报告",
        ),
    ),
    StageDefinition(
        name="C样阶段",
        slug="Stage_C",
        requirements=(
            "更新PFMEA",
            "量产产线开发进展报告",
            "更新样品包装方案",
            "更新过程特殊特性清单",
            "更新CP",
            "更新SOP",
            "工艺验证计划",
            "样品历史问题清单",
            "CMK分析报告",
            "CPK分析报告",
            "工程变更履历表",
            "产品可制造性分析与风险应对报告",
            "更新过程流程图",
            "更新过程特殊特性清单",
            "设备停机率统计表&设备故障记录表",
            "工艺验证报告",
            "外观标准书",
            "PV测试报告",
        ),
    ),
)


STAGE_ORDER: Tuple[str, ...] = tuple(stage.name for stage in STAGES)


STAGE_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    stage.name: stage.requirements for stage in STAGES
}


STAGE_SLUG_MAP: Dict[str, str] = {stage.name: stage.slug for stage in STAGES}


__all__ = [
    "StageDefinition",
    "STAGES",
    "STAGE_ORDER",
    "STAGE_REQUIREMENTS",
    "STAGE_SLUG_MAP",
]
