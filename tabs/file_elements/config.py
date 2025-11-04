"""Static configuration and metadata for文件要素检查."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

Severity = str


@dataclass(frozen=True)
class ElementRequirement:
    """Describe a mandatory element within a deliverable."""

    key: str
    name: str
    severity: Severity
    description: str
    guidance: str
    keywords: Tuple[str, ...]


@dataclass(frozen=True)
class DeliverableProfile:
    """Capture deliverable metadata and its required elements."""

    id: str
    stage: str
    name: str
    description: str
    references: Tuple[str, ...]
    requirements: Tuple[ElementRequirement, ...]


SEVERITY_LABELS: Dict[Severity, str] = {
    "critical": "关键项",
    "major": "重要项",
    "minor": "提示项",
}

SEVERITY_ORDER: Tuple[Severity, ...] = ("critical", "major", "minor")


DELIVERABLE_PROFILES: Tuple[DeliverableProfile, ...] = (
    DeliverableProfile(
        id="project_charter",
        stage="立项阶段",
        name="项目立项报告",
        description="明确立项背景、目标以及资源计划，作为APQP项目启动的基石。",
        references=("AIAG APQP 第2章", "客户特殊要求"),
        requirements=(
            ElementRequirement(
                key="scope",
                name="项目范围界定",
                severity="critical",
                description="定义项目边界、产品范围以及交付的系统或组件。",
                guidance="需覆盖客户、内部相关部门确认的边界条件，并标记超出范围的项目。",
                keywords=("项目范围", "范围界定", "Scope"),
            ),
            ElementRequirement(
                key="objectives",
                name="质量/成本/交付目标",
                severity="critical",
                description="列出质量、成本、交付(QCD)等可量化目标。",
                guidance="建议以SMART原则描述，如一次交付合格率≥98%。",
                keywords=("质量目标", "成本目标", "交付目标", "QCD"),
            ),
            ElementRequirement(
                key="timeline",
                name="里程碑计划",
                severity="major",
                description="给出关键里程碑节点，覆盖A/B/C样与SOP时间。",
                guidance="可采用甘特图或里程碑表的形式，需包含客户关键审核节点。",
                keywords=("里程碑", "Milestone", "时间计划"),
            ),
            ElementRequirement(
                key="risk",
                name="立项风险识别",
                severity="major",
                description="识别项目启动阶段存在的关键风险并提出应对。",
                guidance="至少覆盖技术、资源、供应链三类风险，并列出负责人。",
                keywords=("风险", "应对措施", "RPN"),
            ),
            ElementRequirement(
                key="team",
                name="APQP团队职责",
                severity="minor",
                description="列出跨职能团队成员及角色。",
                guidance="应体现质量、工程、采购、供应商等角色的职责分配。",
                keywords=("团队", "职责", "RASIC", "RACI"),
            ),
        ),
    ),
    DeliverableProfile(
        id="design_fmea",
        stage="A样阶段",
        name="DFMEA",
        description="识别设计风险并规划预防措施，是A样阶段的核心交付物。",
        references=("AIAG&VDA FMEA 手册",),
        requirements=(
            ElementRequirement(
                key="structure",
                name="功能-失效结构树",
                severity="critical",
                description="建立系统-子系统-零部件结构，支撑功能与失效分析。",
                guidance="建议引用结构树或边界图，覆盖客户接口及关键特性。",
                keywords=("结构树", "边界图", "功能结构"),
            ),
            ElementRequirement(
                key="function",
                name="功能描述",
                severity="critical",
                description="明确每个层级的功能需求及性能指标。",
                guidance="需与客户需求矩阵一致，包含量化指标。",
                keywords=("功能", "要求", "性能指标"),
            ),
            ElementRequirement(
                key="failure",
                name="潜在失效模式",
                severity="major",
                description="列出潜在失效模式并关联失效后果。",
                guidance="覆盖历史问题与三新风险，确保后果与严重度匹配。",
                keywords=("失效模式", "Failure Mode", "严重度"),
            ),
            ElementRequirement(
                key="prevention",
                name="预防/探测措施",
                severity="major",
                description="列出当前的预防和探测措施，确保风险可控。",
                guidance="至少包含设计验证、仿真、试验等措施及频次。",
                keywords=("预防措施", "探测措施", "控制计划"),
            ),
            ElementRequirement(
                key="action",
                name="改进措施及责任",
                severity="minor",
                description="针对高风险项制定改进措施、责任人与完成时间。",
                guidance="建议列出截止日期及状态追踪，符合FMEA闭环要求。",
                keywords=("改进措施", "责任人", "完成日期"),
            ),
        ),
    ),
    DeliverableProfile(
        id="process_flow",
        stage="B样阶段",
        name="过程流程图",
        description="描述制造流程与关键控制点，为PFMEA和控制计划提供输入。",
        references=("AIAG APQP 第4章", "VDA-RGA"),
        requirements=(
            ElementRequirement(
                key="scope",
                name="工艺范围覆盖",
                severity="critical",
                description="流程需覆盖从来料到出货的全部工序。",
                guidance="建议标记来料检验、制程、出货检验等关键节点。",
                keywords=("工序", "流程", "范围", "来料", "出货"),
            ),
            ElementRequirement(
                key="characteristic",
                name="特殊特性标识",
                severity="critical",
                description="明确过程中的特殊特性并标注符号。",
                guidance="使用△、◇、※等符号与控制计划保持一致。",
                keywords=("特殊特性", "关键特性", "符号"),
            ),
            ElementRequirement(
                key="parameter",
                name="关键过程参数",
                severity="major",
                description="列出关键过程参数及公差/控制方法。",
                guidance="应包含检测频次、设备、记录方式，满足溯源。",
                keywords=("过程参数", "公差", "控制方法"),
            ),
            ElementRequirement(
                key="inspection",
                name="检验/测试节点",
                severity="major",
                description="识别过程检验和试验节点。",
                guidance="包括进料检验、过程抽检、100%检测等，并写明判定准则。",
                keywords=("检验", "测试", "试验", "抽检"),
            ),
            ElementRequirement(
                key="escalation",
                name="异常处理流程",
                severity="minor",
                description="定义过程异常的响应和升级机制。",
                guidance="应明确隔离、评审、放行步骤及责任人。",
                keywords=("异常", "处置", "隔离", "升级"),
            ),
        ),
    ),
    DeliverableProfile(
        id="control_plan",
        stage="C样阶段",
        name="控制计划",
        description="量产前控制策略的汇总文件，确保过程受控。",
        references=("AIAG 控制计划手册", "IATF 16949 8.5.1"),
        requirements=(
            ElementRequirement(
                key="linkage",
                name="FMEA/流程图联动",
                severity="critical",
                description="控制计划项目应与PFMEA及流程图保持一致。",
                guidance="建议列出PFMEA编号/流程工序号，确保横向追溯。",
                keywords=("PFMEA", "流程图", "编号", "对应"),
            ),
            ElementRequirement(
                key="method",
                name="控制方法与频次",
                severity="critical",
                description="说明控制方法、抽样计划及检查频次。",
                guidance="需标注计量/计数方法、控制界限及记录方式。",
                keywords=("控制方法", "频次", "抽样", "记录"),
            ),
            ElementRequirement(
                key="reaction",
                name="失控反应计划",
                severity="major",
                description="定义失控时的响应、隔离及纠正措施。",
                guidance="应包含停机、隔离、复判、原因分析等步骤。",
                keywords=("失控", "反应计划", "纠正措施"),
            ),
            ElementRequirement(
                key="evidence",
                name="检验记录要求",
                severity="major",
                description="明确检验记录的保留与追溯要求。",
                guidance="建议写明记录格式、保存周期、责任部门。",
                keywords=("记录", "保存", "追溯", "Retention"),
            ),
            ElementRequirement(
                key="approval",
                name="批准签核",
                severity="minor",
                description="控制计划需经跨职能团队批准。",
                guidance="至少包含质量、工程、生产、客户代表的签核信息。",
                keywords=("签核", "批准", "审批", "签名"),
            ),
        ),
    ),
)


PHASE_TO_DELIVERABLES: Dict[str, Tuple[DeliverableProfile, ...]] = {}
for profile in DELIVERABLE_PROFILES:
    PHASE_TO_DELIVERABLES.setdefault(profile.stage, tuple())
PHASE_TO_DELIVERABLES = {
    stage: tuple(profile for profile in DELIVERABLE_PROFILES if profile.stage == stage)
    for stage in {profile.stage for profile in DELIVERABLE_PROFILES}
}


DELIVERABLE_INDEX: Dict[str, DeliverableProfile] = {
    profile.id: profile for profile in DELIVERABLE_PROFILES
}

__all__ = [
    "ElementRequirement",
    "DeliverableProfile",
    "DELIVERABLE_PROFILES",
    "PHASE_TO_DELIVERABLES",
    "DELIVERABLE_INDEX",
    "SEVERITY_LABELS",
    "SEVERITY_ORDER",
]
