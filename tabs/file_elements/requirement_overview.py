"""Requirement overview templates for常见APQP交付物."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

RequirementRow = Dict[str, str]


def _rows(*items: Tuple[str, str, str, str]) -> List[RequirementRow]:
    rows: List[RequirementRow] = []
    for severity, element, description, guidance in items:
        rows.append(
            {
                "severity": severity,
                "element": element,
                "description": description,
                "guidance": guidance,
            }
        )
    return rows


DEFAULT_OVERVIEW = {
    "summary": "参考APQP通用要求，需覆盖目的、输入、输出及责任人。",
    "references": ("AIAG APQP",),
    "requirements": _rows(
        (
            "critical",
            "文件范围与目的",
            "明确文档适用的产品/工艺范围及目标。",
            "给出版本号、适用车型/项目及审批日期。",
        ),
        (
            "major",
            "关键输入",
            "列出编制所依据的客户/内部标准及数据来源。",
            "引用客户图纸、规范或历史数据，并保持可追溯。",
        ),
        (
            "major",
            "输出与交付",
            "说明文档最终输出的结论、指标或执行要求。",
            "使用表格或清单列出关键指标及责任部门。",
        ),
        (
            "minor",
            "审批与维护",
            "描述签核流程及更新频率。",
            "标注签核人、日期，并给出下次复审计划。",
        ),
    ),
}


CATEGORY_OVERVIEWS: Dict[str, Dict[str, object]] = {
    "feasibility": {
        "summary": "从市场、技术、产能、供应链等维度论证项目开展的可行性。",
        "references": ("AIAG APQP 第2章", "公司立项流程"),
        "requirements": _rows(
            (
                "critical",
                "客户/市场需求",
                "分析客户量、价格、法规及竞争态势。",
                "引用客户预测、目标价格及差异。",
            ),
            (
                "critical",
                "技术可行性",
                "评估技术平台、专利与差距。",
                "列出关键技术风险及验证计划。",
            ),
            (
                "major",
                "产能与供应链",
                "说明内部产能、供应商能力及瓶颈。",
                "附产能测算、供应商状态和备选方案。",
            ),
            (
                "minor",
                "财务回报",
                "测算投资、回收期与收益。",
                "列出关键假设与敏感性分析。",
            ),
        ),
    },
    "risk_report": {
        "summary": "识别项目层面风险并制定缓解措施。",
        "references": ("AIAG APQP 第3章", "IATF 16949 6.1"),
        "requirements": _rows(
            (
                "critical",
                "风险清单",
                "覆盖技术、质量、供应链、法规等维度。",
                "使用矩阵或FMEA列出风险项。",
            ),
            (
                "major",
                "评估方法",
                "说明严重度、概率、探测的定义。",
                "标注评分标准、责任人及日期。",
            ),
            (
                "major",
                "缓解措施",
                "针对高风险项制定对策。",
                "列出措施、完成时间及跟踪机制。",
            ),
            (
                "minor",
                "监控复审",
                "定义复审频次与触发条件。",
                "说明如何更新并向管理层汇报。",
            ),
        ),
    },
    "project_plan": {
        "summary": "统筹项目目标、里程碑、资源与沟通机制。",
        "references": ("PMI PMBOK", "AIAG APQP 第2章"),
        "requirements": _rows(
            (
                "critical",
                "项目目标与范围",
                "描述QCD目标及适用范围。",
                "采用SMART指标并标注不在范围项。",
            ),
            (
                "major",
                "里程碑计划",
                "列出A/B/C样、PPAP、SOP等节点。",
                "标注日期、责任人和审批要求。",
            ),
            (
                "major",
                "资源配置",
                "说明人力、设备、模具、供应商需求。",
                "引用RASIC或资源冲突分析。",
            ),
            (
                "minor",
                "沟通机制",
                "定义例会节奏与升级通道。",
                "列出沟通模板及联系人。",
            ),
        ),
    },
    "team_plan": {
        "summary": "明确跨职能团队构成、职责与授权。",
        "references": ("AIAG APQP 第1章",),
        "requirements": _rows(
            (
                "critical",
                "团队名单",
                "列出项目经理及核心职能代表。",
                "包含部门、联系方式和主要职责。",
            ),
            (
                "major",
                "职责划分",
                "定义决策、审核、执行角色。",
                "用RACI/RASIC形式呈现。",
            ),
            (
                "major",
                "资源/授权",
                "说明所需资源和管理授权。",
                "列出关键决策权限与审批人。",
            ),
            (
                "minor",
                "培训与绩效",
                "描述培训计划及绩效指标。",
                "列出资格要求与考核方式。",
            ),
        ),
    },
    "budget_plan": {
        "summary": "规划项目预算、现金流与控制点。",
        "references": ("企业年度预算制度",),
        "requirements": _rows(
            (
                "critical",
                "成本科目",
                "拆分研发、试制、模具、验证、投资。",
                "列出金额、假设及数据来源。",
            ),
            (
                "major",
                "现金流计划",
                "预测阶段性现金流出/流入。",
                "标记支付节点与供应商条件。",
            ),
            (
                "major",
                "预算责任",
                "指定责任部门与审批流程。",
                "说明预警阈值与纠偏机制。",
            ),
            (
                "minor",
                "ROI分析",
                "提供回收期与敏感性分析。",
                "列出关键假设与备选情景。",
            ),
        ),
    },
    "timeline": {
        "summary": "形成跨阶段详细排程与关键路径。",
        "references": ("AIAG APQP 里程碑模板",),
        "requirements": _rows(
            (
                "critical",
                "关键路径",
                "识别影响SOP的关键活动。",
                "标注依赖关系与缓冲策略。",
            ),
            (
                "major",
                "阶段缓冲",
                "说明安全期及设置依据。",
                "给出应急方案与触发条件。",
            ),
            (
                "major",
                "责任分配",
                "各里程碑需对应责任部门。",
                "附交付物、审批人及状态。",
            ),
            (
                "minor",
                "监控机制",
                "定义跟踪频次与版本管理。",
                "说明使用的计划工具。",
            ),
        ),
    },
    "quality_target": {
        "summary": "分解客户质量指标，形成项目内控KPI。",
        "references": ("IATF 16949 6.2",),
        "requirements": _rows(
            (
                "critical",
                "客户指标映射",
                "对照客户PPM、OQD等指标。",
                "列出目标值、数据来源与计算方式。",
            ),
            (
                "major",
                "阶段目标",
                "按样件/量产阶段设定目标。",
                "关联验证活动及责任人。",
            ),
            (
                "major",
                "监控预警",
                "定义监控频次与预警阈值。",
                "引用看板或统计方法。",
            ),
            (
                "minor",
                "改善计划",
                "偏离时的纠偏行动。",
                "列出闭环流程与完成时间。",
            ),
        ),
    },
    "cost_target": {
        "summary": "设定材料、制造、物流等成本控制目标。",
        "references": ("企业成本管理制度",),
        "requirements": _rows(
            (
                "critical",
                "目标分解",
                "拆分材料、制造、物流、质保成本。",
                "引用基准成本与客户目标价。",
            ),
            (
                "major",
                "改善路径",
                "列出成本优化措施。",
                "涵盖设计优化、供应链策略。",
            ),
            (
                "major",
                "监控指标",
                "设定阶段性成本跟踪。",
                "说明测算方法与责任人。",
            ),
            (
                "minor",
                "变更管理",
                "成本变更审批与客户沟通。",
                "关联工程变更与采购谈判。",
            ),
        ),
    },
    "deliverable_matrix": {
        "summary": "列示各阶段交付物、责任与状态。",
        "references": ("AIAG APQP 进度跟踪表",),
        "requirements": _rows(
            (
                "critical",
                "交付物列表",
                "覆盖全部APQP阶段文档与样件。",
                "标注版本、责任部门、状态。",
            ),
            (
                "major",
                "里程碑关联",
                "每项交付物对应具体节点。",
                "列出计划完成时间及验收标准。",
            ),
            (
                "major",
                "输入输出关系",
                "说明文档之间的依赖链。",
                "突出FMEA-CP-SOP联动。",
            ),
            (
                "minor",
                "归档控制",
                "定义存储位置与权限。",
                "提供PLM/共享盘路径与版本策略。",
            ),
        ),
    },
    "cell_spec": {
        "summary": "定义电芯性能、尺寸及测试要求。",
        "references": ("客户技术规范", "UN38.3"),
        "requirements": _rows(
            (
                "critical",
                "电性能指标",
                "列出容量、电压、内阻等指标。",
                "包含测试条件与判退准则。",
            ),
            (
                "critical",
                "尺寸与结构",
                "说明外形尺寸、极耳位置及公差。",
                "引用2D/3D图纸编号。",
            ),
            (
                "major",
                "可靠性要求",
                "规定循环寿命、保存寿命。",
                "列出验证标准与样本量。",
            ),
            (
                "minor",
                "包装与标识",
                "描述标签、条码、安全提示。",
                "与客户规范保持一致并附样例。",
            ),
        ),
    },
    "tolerance_chain": {
        "summary": "验证关键尺寸链累积误差满足装配要求。",
        "references": ("GD&T", "VDA 5"),
        "requirements": _rows(
            (
                "critical",
                "尺寸链定义",
                "明确基准、终点和相关零件。",
                "提供示意图或坐标系。",
            ),
            (
                "critical",
                "计算方法",
                "说明公差叠加方法及假设。",
                "列出统计/最坏情况计算结果。",
            ),
            (
                "major",
                "敏感性分析",
                "识别主导尺寸及其影响。",
                "提出控制策略或设计调整。",
            ),
            (
                "minor",
                "验证计划",
                "定义测量手段与频次。",
                "引用量检具编号与校准状态。",
            ),
        ),
    },
    "special_characteristics": {
        "summary": "识别需要重点控制的产品/过程特性。",
        "references": ("AIAG 特殊特性手册",),
        "requirements": _rows(
            (
                "critical",
                "特性识别逻辑",
                "定义符号、准则及判定流程。",
                "说明与客户符号的对应关系。",
            ),
            (
                "major",
                "清单内容",
                "列出特性描述、位置、控制方法。",
                "关联DFMEA/PFMEA编号。",
            ),
            (
                "major",
                "验证/监控",
                "描述测量频次、设备、记录方式。",
                "提供控制计划或检验卡片引用。",
            ),
            (
                "minor",
                "变更管理",
                "说明新增/取消特性的审批流程。",
                "列出责任人及客户沟通方式。",
            ),
        ),
    },
    "three_new": {
        "summary": "梳理新材料、新工艺、新设备清单及风险。",
        "references": ("企业三新管理规范",),
        "requirements": _rows(
            (
                "critical",
                "三新条目",
                "列出涉及的新材料/工艺/设备。",
                "提供供应商、状态及应用范围。",
            ),
            (
                "major",
                "验证与准入",
                "说明验证项目、样本、判定准则。",
                "引用验证计划或试验报告。",
            ),
            (
                "major",
                "风险评估",
                "识别对质量、交付的影响。",
                "列出控制措施与责任人。",
            ),
            (
                "minor",
                "状态跟踪",
                "记录当前进度与下一步动作。",
                "保持与采购/供应商共享。",
            ),
        ),
    },
    "process_standard": {
        "summary": "定义工序操作、参数和质量控制要求。",
        "references": ("IATF 16949 8.5.1",),
        "requirements": _rows(
            (
                "critical",
                "工序定义",
                "描述工序目的、设备、材料。",
                "附工序编号和流程位置。",
            ),
            (
                "major",
                "作业参数",
                "列出关键参数及允差。",
                "标注测量方法、记录表。",
            ),
            (
                "major",
                "质量控制",
                "说明检验项目、频次、反应计划。",
                "对应控制计划/作业指导书。",
            ),
            (
                "minor",
                "安全与5S",
                "注明安全注意事项与5S要求。",
                "提供图示或培训记录。",
            ),
        ),
    },
    "tooling_list": {
        "summary": "列出项目涉及的模具/工装及状态。",
        "references": ("模具管理规范",),
        "requirements": _rows(
            (
                "critical",
                "模具清单",
                "包含模具编号、用途、供应商。",
                "标注位置、计划交期。",
            ),
            (
                "major",
                "状态跟踪",
                "记录设计、制造、验收进度。",
                "注明问题、责任人与恢复计划。",
            ),
            (
                "major",
                "资产/成本",
                "列出投资额、摊销计划或结算方式。",
                "说明所有权及折旧策略。",
            ),
            (
                "minor",
                "维护计划",
                "描述保养周期与责任部门。",
                "可附保养记录模板。",
            ),
        ),
    },
    "cad_3d": {
        "summary": "提供产品三维模型以支持设计、仿真与制造。",
        "references": ("CAD 数据管理规范",),
        "requirements": _rows(
            (
                "critical",
                "模型版本",
                "标注软件、版本号与发布日期。",
                "需与BOM/图纸同步。",
            ),
            (
                "major",
                "特征定义",
                "确保关键特征、接口完整。",
                "提供图层/命名规则说明。",
            ),
            (
                "major",
                "引用/链接",
                "说明模型与下游仿真/加工的接口。",
                "附STEP/IGES等导出格式。",
            ),
            (
                "minor",
                "数据安全",
                "定义访问权限与加密措施。",
                "记录发布审批与分发范围。",
            ),
        ),
    },
    "drawing_2d": {
        "summary": "2D工程图传达尺寸、公差和检验要求。",
        "references": ("ISO 1101", "ASME Y14.5"),
        "requirements": _rows(
            (
                "critical",
                "尺寸/公差",
                "列出关键尺寸、公差、基准。",
                "包含GD&T符号及说明。",
            ),
            (
                "major",
                "材料/处理",
                "注明材料、热处理、表面处理。",
                "与BOM/工艺路线一致。",
            ),
            (
                "major",
                "检验指引",
                "指明测量方法与检验等级。",
                "附量检具编号或备注。",
            ),
            (
                "minor",
                "修订记录",
                "显示版本、变更内容、签核。",
                "保持与PLM数据一致。",
            ),
        ),
    },
    "bom": {
        "summary": "BOM清单定义零部件结构与用量。",
        "references": ("EBOM/SBOM 管理规范",),
        "requirements": _rows(
            (
                "critical",
                "层级结构",
                "展示父子零件关系与数量。",
                "标注物料号、描述、版本。",
            ),
            (
                "major",
                "属性字段",
                "包含材料、供应商、替代件信息。",
                "说明关键受控件及审批状态。",
            ),
            (
                "major",
                "变更控制",
                "记录BOM更改原因与生效批次。",
                "关联ECN/PCN号。",
            ),
            (
                "minor",
                "导出与共享",
                "说明数据导出格式与共享渠道。",
                "保持与ERP/PLM同步。",
            ),
        ),
    },
    "simulation": {
        "summary": "仿真报告验证设计在多物理场下的表现。",
        "references": ("CAE 验证规范",),
        "requirements": _rows(
            (
                "critical",
                "工况与边界",
                "描述模型假设、载荷、约束。",
                "与真实工况一致并注明来源。",
            ),
            (
                "major",
                "结果与结论",
                "展示关键指标与是否满足要求。",
                "附图形/表格与判定标准。",
            ),
            (
                "major",
                "验证/相关性",
                "说明与试验结果或经验数据的对比。",
                "列出差异及改进措施。",
            ),
            (
                "minor",
                "模型版本管理",
                "记录软件版本与输入文件。",
                "提供可追溯的存储路径。",
            ),
        ),
    },
    "test_master_plan": {
        "summary": "测试大纲规划验证项目、样本及判定准则。",
        "references": ("DV/PV 测试规范",),
        "requirements": _rows(
            (
                "critical",
                "测试矩阵",
                "列出所有测试项目、标准与目的。",
                "按阶段区分DV、PV、量产确认。",
            ),
            (
                "major",
                "样本/批次",
                "定义样本量、批次来源、状态。",
                "注明是否包含三新件或极限件。",
            ),
            (
                "major",
                "判定准则",
                "提供合格判定、允许偏差。",
                "引用客户/行业标准编号。",
            ),
            (
                "minor",
                "进度与责任",
                "列出计划时间、实验室、责任人。",
                "说明报告提交和偏差处理流程。",
            ),
        ),
    },
    "patent_mining": {
        "summary": "专利挖掘清单识别潜在专利点与申请计划。",
        "references": ("企业知识产权管理规范",),
        "requirements": _rows(
            (
                "major",
                "创新点描述",
                "列出可申请的技术方案与优势。",
                "包含技术背景、现有技术对比。",
            ),
            (
                "major",
                "专利策略",
                "定义申请类型、区域、时间表。",
                "注明权利要求草案或关键词。",
            ),
            (
                "minor",
                "责任与状态",
                "指定技术与法务责任人。",
                "记录撰写、递交、审查进度。",
            ),
            (
                "minor",
                "风险提示",
                "评估侵权风险或保密要求。",
                "列出需回避的已授权专利。",
            ),
        ),
    },
    "pfmea": {
        "summary": "PFMEA识别过程风险并制定控制措施。",
        "references": ("AIAG&VDA FMEA 手册",),
        "requirements": _rows(
            (
                "critical",
                "工序结构与功能",
                "列出工序、功能及对应特性。",
                "与流程图、控制计划一致。",
            ),
            (
                "major",
                "失效模式分析",
                "识别潜在失效、后果、原因。",
                "提供S/O/D评价与依据。",
            ),
            (
                "major",
                "控制与改进",
                "列出现有控制、改进措施、责任人。",
                "注明完成期限与验证方式。",
            ),
            (
                "minor",
                "版本与跟踪",
                "保持版本记录及变更原因。",
                "说明触发重新分析的条件。",
            ),
        ),
    },
    "line_layout": {
        "summary": "产线规划方案确定布置、产能与物流路径。",
        "references": ("精益布局指南",),
        "requirements": _rows(
            (
                "critical",
                "产线配置",
                "描述工序顺序、设备、节拍。",
                "附布置图及产能计算。",
            ),
            (
                "major",
                "物流与人机工程",
                "规划物料流、WIP、工位人因。",
                "识别瓶颈并提出改善。",
            ),
            (
                "major",
                "设施需求",
                "列出公用工程、治具、IT接口。",
                "说明投资、安装计划。",
            ),
            (
                "minor",
                "柔性与扩展",
                "评估未来扩产、切换能力。",
                "列出模块化方案或空位。",
            ),
        ),
    },
    "process_design": {
        "summary": "过程设计方案定义关键工艺路线与能力要求。",
        "references": ("AIAG APQP 第4章",),
        "requirements": _rows(
            (
                "critical",
                "工艺路线",
                "描述从原料到成品的主要工序。",
                "标注关键参数与设备。",
            ),
            (
                "major",
                "能力需求",
                "给出节拍、良率、能力指标。",
                "说明需验证的关键工序。",
            ),
            (
                "major",
                "资源规划",
                "列出人力、设备、治具、IT。",
                "包含建设周期与风险。",
            ),
            (
                "minor",
                "质量策划",
                "说明与FMEA、控制计划的接口。",
                "标注所需文件和负责部门。",
            ),
        ),
    },
    "manufacturability": {
        "summary": "评估设计可制造性并提出风险应对。",
        "references": ("DFM/A 审查指南",),
        "requirements": _rows(
            (
                "critical",
                "可制造性分析",
                "识别设计对制程的挑战。",
                "列出结构、材料、装配等问题。",
            ),
            (
                "major",
                "成本/效率影响",
                "评估设计对成本、节拍影响。",
                "提供量化数据或模拟结果。",
            ),
            (
                "major",
                "风险与对策",
                "针对关键风险提出解决方案。",
                "明确责任人和完成计划。",
            ),
            (
                "minor",
                "跟踪机制",
                "说明整改闭环与复核节点。",
                "关联设计变更和量产验证。",
            ),
        ),
    },
    "process_flow": {
        "summary": "过程流程图描述制造流程与控制点。",
        "references": ("AIAG APQP 第4章",),
        "requirements": _rows(
            (
                "critical",
                "范围覆盖",
                "覆盖来料、加工、检验、出货。",
                "按工序编号及物料流展示。",
            ),
            (
                "major",
                "特殊特性",
                "标注关键/特殊特性符号。",
                "与控制计划、FMEA一致。",
            ),
            (
                "major",
                "检测节点",
                "说明检验/测试工序与记录。",
                "标注样本量、频次、标准。",
            ),
            (
                "minor",
                "异常处理",
                "描述异常响应与升级流程。",
                "包含隔离、复判、放行步骤。",
            ),
        ),
    },
    "process_specials": {
        "summary": "过程特殊特性清单聚焦工序控制重点。",
        "references": ("特殊特性控制程序",),
        "requirements": _rows(
            (
                "critical",
                "特性定义",
                "列出工序特性、符号、规格。",
                "对应PFMEA风险与控制计划。",
            ),
            (
                "major",
                "控制方法",
                "描述测量方法、频次、记录。",
                "给出量具、CP/CPK要求。",
            ),
            (
                "major",
                "反应计划",
                "失控时的隔离、返工、通知流程。",
                "引用SOP或应急预案。",
            ),
            (
                "minor",
                "版本管理",
                "说明新增/移除特性的审批。",
                "列出客户沟通记录。",
            ),
        ),
    },
    "control_plan": {
        "summary": "控制计划汇总过程控制策略与反应计划。",
        "references": ("AIAG 控制计划手册",),
        "requirements": _rows(
            (
                "critical",
                "FMEA/流程图联动",
                "确保控制计划字段与PFMEA、流程图一致。",
                "列出工序号、特性、PFMEA编号。",
            ),
            (
                "major",
                "控制方法与频次",
                "描述测量/监控方法、频次、记录。",
                "说明样本量、量具、记录表。",
            ),
            (
                "major",
                "反应计划",
                "定义失控时的隔离、纠正、通知。",
                "引用SOP或异常处理流程。",
            ),
            (
                "minor",
                "审批与维护",
                "列出批准人、日期及更新机制。",
                "指明与客户沟通的要求。",
            ),
        ),
    },
    "sop": {
        "summary": "SOP/作业指导书指导现场操作与质量控制。",
        "references": ("IATF 16949 8.5.1",),
        "requirements": _rows(
            (
                "critical",
                "操作步骤",
                "按顺序描述操作内容与注意事项。",
                "配合图片/图标辅助理解。",
            ),
            (
                "major",
                "关键参数",
                "列出需记录或确认的参数。",
                "引用控制计划或检查表。",
            ),
            (
                "major",
                "安全/防错",
                "说明安全要求、防错措施。",
                "提供Poka-Yoke或警示信息。",
            ),
            (
                "minor",
                "培训与签核",
                "记录培训日期、受训人及批准人。",
                "注明文件位置与版本。",
            ),
        ),
    },
    "process_validation_plan": {
        "summary": "工艺验证计划定义量产前验证策略与资源。",
        "references": ("PPAP 指南",),
        "requirements": _rows(
            (
                "critical",
                "验证范围",
                "列出需验证的工序、设备、特性。",
                "标注判定标准与样本。",
            ),
            (
                "major",
                "计划与日程",
                "提供验证时间表及责任人。",
                "涵盖试运行、能力验证、耐久测试。",
            ),
            (
                "major",
                "资源与设备",
                "说明所需工装、量具、实验室。",
                "附计量状态和备份方案。",
            ),
            (
                "minor",
                "结果输出",
                "定义报告、签核及问题处理流程。",
                "列出数据收集模板或IT系统。",
            ),
        ),
    },
    "packaging": {
        "summary": "样品包装方案确保运输安全与合规。",
        "references": ("客户物流规范",),
        "requirements": _rows(
            (
                "critical",
                "包装结构",
                "描述内外包装材料、结构。",
                "附图示、堆码方式、尺寸。",
            ),
            (
                "major",
                "防护措施",
                "说明防静电、防潮、防震等措施。",
                "列出缓冲材料及验证结果。",
            ),
            (
                "major",
                "标签与识别",
                "定义标签内容、条码、危险标识。",
                "满足客户/法规要求。",
            ),
            (
                "minor",
                "循环/环保",
                "说明循环包装或环保合规。",
                "提供回收流程与责任人。",
            ),
        ),
    },
    "change_log": {
        "summary": "变更履历表记录设计/工程变更的全过程。",
        "references": ("ECN/PCN 管理流程",),
        "requirements": _rows(
            (
                "critical",
                "变更信息",
                "列出变更编号、原因、范围。",
                "对应客户通知与受影响物料。",
            ),
            (
                "major",
                "评估与验证",
                "记录风险评估、验证计划、结果。",
                "链接FMEA/控制计划更新。",
            ),
            (
                "major",
                "批准与通知",
                "显示审批人、日期、客户确认。",
                "列出通知范围及执行状态。",
            ),
            (
                "minor",
                "实施追踪",
                "标注生效批次、库存切换计划。",
                "附闭环确认记录。",
            ),
        ),
    },
    "dv_report": {
        "summary": "DV测试报告验证设计满足性能要求。",
        "references": ("DV&PV Test Plan",),
        "requirements": _rows(
            (
                "critical",
                "测试概述",
                "说明目的、样本、条件。",
                "引用测试标准及偏差。",
            ),
            (
                "major",
                "结果与判定",
                "呈现数据、图像、结论。",
                "对比目标并说明是否通过。",
            ),
            (
                "major",
                "问题与措施",
                "记录不合格项、根因、纠正。",
                "提供闭环状态及责任人。",
            ),
            (
                "minor",
                "附件与追溯",
                "附原始记录、照片、日志。",
                "标注文件编号与存档位置。",
            ),
        ),
    },
    "line_progress": {
        "summary": "量产产线开发进展报告跟踪产线建设状态。",
        "references": ("量产准备清单",),
        "requirements": _rows(
            (
                "critical",
                "进度概览",
                "列出各工段完成度与关键节点。",
                "对比计划与实际，解释偏差。",
            ),
            (
                "major",
                "问题与风险",
                "识别影响投产的障碍。",
                "提供应对措施及责任人。",
            ),
            (
                "major",
                "资源状态",
                "更新设备、治具、人员、IT准备情况。",
                "说明待采购或待验收项目。",
            ),
            (
                "minor",
                "下一步计划",
                "列出近期重点、所需支持。",
                "明确升级需求和决策时间。",
            ),
        ),
    },
    "issue_history": {
        "summary": "样品历史问题清单记录遗留问题及关闭状态。",
        "references": ("Lessons Learned 程序",),
        "requirements": _rows(
            (
                "critical",
                "问题描述",
                "清楚描述发生阶段、现象、影响。",
                "引用编号、照片或数据。",
            ),
            (
                "major",
                "根因与措施",
                "记录分析结论、临时/永久措施。",
                "标注责任人与完成日期。",
            ),
            (
                "major",
                "验证效果",
                "说明验证方法、结果、证据。",
                "与客户/内部确认保持一致。",
            ),
            (
                "minor",
                "经验沉淀",
                "总结教训及在后续阶段的应用。",
                "链接标准更新或培训。",
            ),
        ),
    },
    "cmk_report": {
        "summary": "CMK报告验证量测系统及设备重复性。",
        "references": ("MSA 手册",),
        "requirements": _rows(
            (
                "critical",
                "测量对象",
                "描述量具/设备、特性、测量范围。",
                "与控制计划一致。",
            ),
            (
                "major",
                "试验方法",
                "说明样本数、次数、操作员。",
                "满足MSA公式要求。",
            ),
            (
                "major",
                "结果评估",
                "给出CMK数值及判定。",
                "提供趋势图或柱状图。",
            ),
            (
                "minor",
                "改进建议",
                "若不合格，提出改进措施。",
                "标注责任人及完成时间。",
            ),
        ),
    },
    "cpk_report": {
        "summary": "CPK报告展示制程能力是否满足要求。",
        "references": ("SPC 手册",),
        "requirements": _rows(
            (
                "critical",
                "数据收集",
                "说明样本量、采集时间、工位。",
                "确保数据代表性。",
            ),
            (
                "major",
                "统计结果",
                "提供均值、标准差、CP/CPK。",
                "展示控制图与判定。",
            ),
            (
                "major",
                "超差分析",
                "识别失控点、原因。",
                "提出纠正措施或加严控制。",
            ),
            (
                "minor",
                "复测计划",
                "列出跟踪频次及责任人。",
                "与控制计划联动。",
            ),
        ),
    },
    "equipment_downtime": {
        "summary": "设备停机率统计与故障记录用于可靠性分析。",
        "references": ("TPM 管理手册",),
        "requirements": _rows(
            (
                "critical",
                "停机统计",
                "记录停机时间、次数、原因。",
                "计算OEE或停机率。",
            ),
            (
                "major",
                "故障分析",
                "描述故障模式、根因。",
                "引用维修记录与备件信息。",
            ),
            (
                "major",
                "改善计划",
                "列出预防/纠正措施。",
                "标注负责人、完成时间。",
            ),
            (
                "minor",
                "数据趋势",
                "展示周/月趋势及预警阈值。",
                "指导TPM活动和备件策略。",
            ),
        ),
    },
    "process_validation_report": {
        "summary": "工艺验证报告汇总试生产、能力验证结果。",
        "references": ("PPAP 手册",),
        "requirements": _rows(
            (
                "critical",
                "验证范围与条件",
                "描述试生产批次、设备、环境。",
                "对应验证计划。",
            ),
            (
                "major",
                "结果与结论",
                "提供关键指标、能力、缺陷数据。",
                "判定是否满足量产条件。",
            ),
            (
                "major",
                "问题与整改",
                "列出发现的问题及闭环状态。",
                "附照片、记录或数据。",
            ),
            (
                "minor",
                "签核与放行",
                "记录批准人、日期与条件。",
                "说明是否需再验证。",
            ),
        ),
    },
    "appearance_spec": {
        "summary": "外观标准书定义可接受的外观缺陷与判定。",
        "references": ("客户外观检验规范",),
        "requirements": _rows(
            (
                "critical",
                "缺陷定义",
                "通过图片/示例描述缺陷类型与等级。",
                "标注判定限值、测量方法。",
            ),
            (
                "major",
                "检验条件",
                "说明光源、距离、角度、检验时间。",
                "与客户标准一致。",
            ),
            (
                "major",
                "判定与分级",
                "给出A/B/C类缺陷及处理方式。",
                "定义返工、报废或特采流程。",
            ),
            (
                "minor",
                "样件/参考",
                "提供标准样件或照片编号。",
                "说明保管位置与更新策略。",
            ),
        ),
    },
    "pv_report": {
        "summary": "PV测试报告验证产品在量产配置下的可靠性。",
        "references": ("DV&PV Test Plan",),
        "requirements": _rows(
            (
                "critical",
                "测试配置",
                "描述量产件、工装、软件版本。",
                "说明样本代表性。",
            ),
            (
                "major",
                "耐久/环境结果",
                "展示关键寿命、环境、可靠性结果。",
                "对照目标值并解释异常。",
            ),
            (
                "major",
                "改进与复测",
                "列出失败项、改进措施、复测安排。",
                "与客户签署的偏差保持一致。",
            ),
            (
                "minor",
                "结论与建议",
                "总结量产准备程度及后续要求。",
                "给出放行条件或需跟踪项目。",
            ),
        ),
    },
}

DELIVERABLE_CATEGORY_MAP: Dict[str, str] = {
    "项目立项报告": "project_plan",
    "项目可行性分析报告": "feasibility",
    "项目风险评估报告": "risk_report",
    "项目计划书": "project_plan",
    "项目团队组建方案": "team_plan",
    "项目预算方案": "budget_plan",
    "项目时间计划": "timeline",
    "项目质量目标": "quality_target",
    "项目成本目标": "cost_target",
    "项目交付物清单": "deliverable_matrix",
    "电芯规格书": "cell_spec",
    "尺寸链公差计算书": "tolerance_chain",
    "DFMEA": "pfmea",
    "PFMEA": "pfmea",
    "特殊特性清单": "special_characteristics",
    "过程特殊特性清单": "process_specials",
    "三新清单": "three_new",
    "制程标准": "process_standard",
    "开模清单": "tooling_list",
    "3D数模": "cad_3d",
    "2D图纸": "drawing_2d",
    "BOM清单": "bom",
    "仿真报告": "simulation",
    "测试大纲": "test_master_plan",
    "专利挖掘清单": "patent_mining",
    "产线规划方案": "line_layout",
    "过程设计初始方案": "process_design",
    "过程设计方案": "process_design",
    "产品可制造性分析及风险应对报告": "manufacturability",
    "产品可制造性分析与风险应对报告": "manufacturability",
    "初始过程流程图": "process_flow",
    "过程流程图": "process_flow",
    "过程特殊特性": "process_specials",
    "初始过程特殊特性": "process_specials",
    "初版CP": "control_plan",
    "控制计划": "control_plan",
    "CP": "control_plan",
    "初版SOP": "sop",
    "SOP": "sop",
    "工艺验证计划": "process_validation_plan",
    "样品包装方案": "packaging",
    "设计变更履历表": "change_log",
    "工程变更履历表": "change_log",
    "DV测试报告": "dv_report",
    "仿真验证报告": "simulation",
    "量产产线开发进展报告": "line_progress",
    "样品历史问题清单": "issue_history",
    "CMK分析报告": "cmk_report",
    "CPK分析报告": "cpk_report",
    "设备停机率统计表&设备故障记录表": "equipment_downtime",
    "工艺验证报告": "process_validation_report",
    "外观标准书": "appearance_spec",
    "PV测试报告": "pv_report",
    "样品包装方案草案": "packaging",
    "三新评估清单": "three_new",
}

PREFIX_CONTEXT: Dict[str, str] = {
    "更新": "update",
    "初始": "initial",
    "初版": "initial",
}


def get_deliverable_overview(deliverable_name: str | None) -> Dict[str, object]:
    """Return summary、参考文献及要素清单."""

    if not deliverable_name:
        overview = DEFAULT_OVERVIEW
        return {
            "summary": overview["summary"],
            "references": overview["references"],
            "requirements": deepcopy(overview["requirements"]),
        }

    base_name = deliverable_name.strip()
    context: str | None = None
    for prefix, ctx in PREFIX_CONTEXT.items():
        if base_name.startswith(prefix) and len(base_name) > len(prefix):
            base_name = base_name[len(prefix) :].strip()
            context = ctx
            break

    category_key = DELIVERABLE_CATEGORY_MAP.get(base_name, base_name)
    overview = CATEGORY_OVERVIEWS.get(category_key, DEFAULT_OVERVIEW)
    requirements = deepcopy(overview.get("requirements", DEFAULT_OVERVIEW["requirements"]))

    if context == "update":
        for row in requirements:
            row["description"] += "（需体现本次修订范围及与上版差异）"
            row["guidance"] += "，同步记录版本号、审批人及生效日期"
    elif context == "initial":
        for row in requirements:
            row["description"] += "（首版基线要求）"
            row["guidance"] += "，确保覆盖客户输入并完成跨部门评审"

    return {
        "summary": overview.get("summary", DEFAULT_OVERVIEW["summary"]),
        "references": overview.get("references", DEFAULT_OVERVIEW["references"]),
        "requirements": requirements,
    }


__all__ = ["get_deliverable_overview"]
