import streamlit as st
from pathlib import Path

def render_help_documentation_tab(session_id):
    """Render the help documentation tab."""
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    
    
    # Create two columns: sidebar navigation and main content
    col1, col2 = st.columns([1, 3])

    # Define section mappings (label -> key)
    section_mappings = {
        "📁 文件齐套性检查": "file_completeness_check",
        "🔍 特殊特性符号检查": "special_symbols_check",
        "📊 设计制程检查": "parameters_check",
        "✅ 文件要素检查": "file_elements_check",
        "🏢 企业标准检查": "enterprise_standard_check",
        "📋 历史问题规避": "history_issues_avoidance",
        "🤖 AI智能体": "ai_agent",
        "⚙️ 设置": "settings",
        "❓ 常见问题": "faq",
        "📚 技术文档": "technical_docs",
    }

    with col1:
        # Minimal nav styling to resemble docs sidebar
        st.markdown(
            """
            <style>
            .help-nav a, .help-nav button {
                width: 100%;
                text-align: left;
            }
            .help-nav .nav-item {
                padding: 6px 8px;
                margin: 2px 0;
                border-radius: 6px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 目录")

        # Current selected section in session_state (default to 📁 文件齐套性检查)
        selected_section = st.session_state.get(
            f"help_section_active_{session_id}", "📁 文件齐套性检查"
        )
        # Backward compatibility: if an old value without icon is stored, reset to default
        valid_labels = set(section_mappings.keys())
        if selected_section not in valid_labels:
            selected_section = "📁 文件齐套性检查"
            st.session_state[f"help_section_active_{session_id}"] = selected_section

        # Render a vertical list of buttons
        for label, key_name in section_mappings.items():
            is_active = (label == selected_section)
            button_label = f"{label}"
            if st.button(
                button_label,
                key=f"help_nav_{key_name}_{session_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state[f"help_section_active_{session_id}"] = label
                st.rerun()

    with col2:
        # Display content based on selection
        if selected_section == "📁 文件齐套性检查":
            render_file_completeness_check_section()
        elif selected_section == "🔍 特殊特性符号检查":
            render_special_symbols_check_section()
        elif selected_section == "📊 设计制程检查":
            render_parameters_check_section()
        elif selected_section == "✅ 文件要素检查":
            render_file_elements_check_section()
        elif selected_section == "🏢 企业标准检查":
            render_enterprise_standard_check_section()
        elif selected_section == "📋 历史问题规避":
            render_history_issues_avoidance_section()
        elif selected_section == "🤖 AI智能体":
            render_ai_agent_section()
        elif selected_section == "⚙️ 设置":
            render_settings_section()
        elif selected_section == "❓ 常见问题":
            render_faq_section()
        elif selected_section == "📚 技术文档":
            render_technical_docs_section(session_id)

def render_special_symbols_check_section():
    """Render the special symbols check section."""
    st.header("🔍 特殊特性符号检查")

    st.markdown(
        """
        特殊特性符号检查已全面升级，现与“🏢 企业标准检查”共用统一的后台工作流：

        - 前端仅负责文件管理与进度展示，所有提示词拼接、检索和大模型调用均在后台完成，避免了超长请求；
        - 支持 **基准文件** 与 **待检查文件** 的分目录管理，上传、删除、清理操作与企业标准页保持一致；
        - 作业会在后台进程中执行，可随时暂停 / 恢复，并在完成后生成 CSV、XLSX 等结构化结果。

        ### 使用流程
        1. 上传或清理右侧文件面板中的基准文件、待检查文件（支持 PDF、Office、文本及压缩包自动解包）。
        2. 点击【开始】：系统会预处理文本、同步基准文件知识库，并按段调用 Bisheng 流程比对符号标注。
        3. 页面实时显示每段提示词与回复的流式内容，可在必要时暂停或停止任务。
        4. 任务完成后，到右侧“分析结果”下载汇总的 CSV/XLSX、对话记录及检查过程文件。

        ### 演示模式
        - 点击【演示】按钮即可复制预置样例，无需等待后台运算即可体验流式展示与结果下载。

        ### 温馨提示
        - 若未上传待检查文本，任务会直接结束并提示“未发现待检查文本”。
        - 基准文件文本会被同步至专用知识库目录，以便 Bisheng 在符号比对时引用出处。
        - “退出登录”按钮仍在后续调优计划中，如需记录相关改动请参阅《how_to_mirror_enterprise.md》。
        """
    )

def render_enterprise_standard_check_section():
    """Render the enterprise standard check help section."""
    st.header("🏢 企业标准检查")

    st.markdown(
        """
        企业标准检查用于将“待检查文件”与“企业标准文件”逐条对照，自动识别不一致项并形成可下载的汇总结果。

        ### 使用流程

        1. 登录并进入本功能页，确认当前用户会话。
        2. 上传文件：
           - 企业标准文件（支持：PDF、Word/PPT、Excel、文本类）
           - 待检查文件（支持：PDF、Word/PPT、Excel、文本类）
        3. 点击【开始】：
           - 系统会清理旧结果，仅处理新增文件；
           - PDF 走 MinerU 解析；Word/PPT 走 Unstructured；Excel 逐表转文本；纯文本类（csv/tsv/md/txt/json/yaml/yml/log/ini/cfg/rst）直接复制为 .txt；
           - 生成的 .txt 分别输出到 `standards_txt` / `examined_txt`；
           - 对“待检查文本”进行分片，比对结果以流式方式展示。
        4. 结构化与导出：
           - 自动生成带中文 JSON 指令的 `prompted_response_*_ptN.txt`；
           - 调用内网 Ollama 进行结构化抽取，产出 `json_*_ptN.txt`；
           - 汇总导出 `CSV/XLSX`（列：技术文件名、技术文件内容、企业标准、不一致之处、理由），并提供下载按钮；
           - 另可生成“企标检查分析过程_yyyymmdd_hhmmss.docx”，包含各文件全部提示词与回复过程（含简易表格清洗与标题层级）。
        5. 【演示】模式：
           - 复制预制样例（含 prompted/json/final_results），直接展示流式内容与下载按钮，无需实时计算。
        
        ### 右侧文件/结果面板
        - 企业标准文件、待检查文件：可查看/删除已上传文件。
        - 分析结果：列出 `final_results` 下的 CSV/XLSX/Word 等文件，支持逐个下载或删除。

        ### 注意事项
        - 未被支持或未解析成功的上传文件，将不会生成对应的 `.txt`，不会进入比对。
        - 若“待检查文本”未生成任何 `.txt`，系统会提示跳过此次比对。
        - 文本同步和清理逻辑：已存在且非空的输出文本跳过；不在上传清单内的孤立文本文件会被清理。

        ### 支持的文件类型（核心）
        - PDF（MinerU）
        - Word/PPT（Unstructured）
        - Excel（逐表转文本）
        - 纯文本类：csv/tsv/md/txt/json/yaml/yml/log/ini/cfg/rst（直接文本化）

        如需扩展更多格式（例如 .odt、.html/.xml、.zip 解包递归处理等），请联系：周昭坤。
        """
    )

def render_parameters_check_section():
    """Render the parameters check section."""
    st.header("⚙️ 设计制程检查")
    
    st.markdown("""
    设计制程检查功能用于验证产品设计参数和制造工艺参数的合理性，确保设计到制造的转换过程符合质量要求。
    
    ### 使用流程
    
    1. 登录并进入本功能页，确认当前用户会话。
    2. 上传文件：
       - 控制计划文件（必选）
       - 待检查文件（目标文件，必选）
       - 图纸文件（可选，用于参考）
    3. 运行方式：
       - 按下【开始】：系统会校验已上传文件并清理旧结果，随后对“您上传的文件”执行分析流程。
       - 按下【演示】：系统会自动拷贝一套示例文件到您的会话目录，并立即运行相同流程，便于快速上手体验。
    4. 系统自动执行三步分析（用语友好版）：
       - 第一步：AI 助手读取并理解各文件中的参数表格，将“控制计划”和“目标文件”的参数内容提取为结构化数据，方便后续逐项对比。
       - 第二步：生成“参数一致性评审结论”，在页面右侧以流式方式展示，直观说明哪些参数一致、哪些存疑或缺失。
       - 第三步：进一步提炼“仅不一致项”，自动整理为可下载/留档的表格，并直接在页面展示预览，便于评审与传阅。
    5. 在页面右侧查看表格与下载结果。如需重跑，可点击【重新开始】。
    
    ### 📁 支持的文件类型
    
    - 控制计划文件、待检查文件：Excel（.xlsx / .xls）
    - 图纸文件：PDF（.pdf）
    
    如需支持更多格式，请联系：周昭坤。
    """)

def render_file_elements_check_section():
    """Render the file elements check section."""
    st.header("📋 文件要素检查")
    
    st.markdown("""
    文件要素检查用于检查APQP文档的完整性和规范性，确保所有必要的要素都已包含。
    
    ### 检查要素
    
    **📄 文档结构**
    - 标题和章节完整性
    - 编号和格式规范性
    - 引用和链接有效性
    
    **📋 内容要素**
    - 必要信息的完整性
    - 数据格式的正确性
    - 逻辑关系的一致性
    
    **🔍 质量要素**
    - 特殊特性标识
    - 控制要求明确性
    - 测量方法描述
    
    ### 检查标准
    
    系统基于以下标准进行检查：
    
    - **APQP标准**: 符合汽车行业APQP要求
    - **ISO标准**: 遵循相关ISO质量管理标准
    - **企业标准**: 符合企业内部质量要求
    - **行业最佳实践**: 基于行业经验总结
    
    注意：
    - 当前为预览版，功能仍在完善中；正式版预计10月上线。
    """)

def render_file_completeness_check_section():
    """Render the file completeness check section."""
    st.header("📁 文件齐套性检查")
    
    st.markdown("""
    文件齐套性检查确保APQP项目所需的所有文档都已准备就绪，没有遗漏。
    
    ### 使用流程

    1. 登录并进入本功能页，确认当前用户会话。
    2. 上传文件：
       - 若右侧已列出历史文件，可先点击【清空所有文件】按钮，以空白环境开始。
       - 按阶段上传：立项阶段、A样阶段、B样阶段、C样阶段分别点击对应上传框添加文件。
       - 右侧“文件管理器”可随时查看/删除已上传文件，并在每个阶段下方“上传新文件”处补充文件。
    3. 运行方式：
       - 若右侧已列出历史文件，可先点击【清空所有文件】按钮，以空白环境开始。
       - 按下【开始】：系统会校验并读取当前阶段文件夹中的文件，清理旧结果后，直接对“您上传的文件”执行齐套性分析。
       - 按下【演示】：系统会自动拷贝一套示例文件到您的会话目录，并立即运行相同流程，便于快速上手体验。
    4. 系统自动执行分析（用语友好版）：
       - 第一步：按APQP阶段读取“应包含的交付物文件清单”，与对应阶段文件夹中的实际文件进行名称近似匹配，对每一项给出“存在/不存在”判断以及匹配到的文件名。
       - 第二步：将多阶段结果统一整理为标准化表格，包含列：Stage、Deliverable、Exists、FileName、Notes，并在页面右侧直接展示预览。
       - 第三步：导出可下载的Excel文件，便于评审、归档与传阅。
    5. 在页面右侧查看表格与下载的Excel。如需重跑，可点击【重新开始】。
    
    ### 🎯 检查清单
    
    以下是默认清单，如果需要换成不同的清单，可以在页面上传。
    - **立项阶段**: 项目立项报告、项目可行性分析报告、项目风险评估报告、项目计划书、项目团队组建方案、项目预算方案、项目时间计划、项目质量目标、项目成本目标、项目交付物清单
    - **A样阶段**: 电芯规格书、尺寸链公差计算书、初始DFMEA、初始特殊特性清单、三新清单、制程标准、开模清单、3D数模、2D图纸、BOM清单、仿真报告、测试大纲、专利挖掘清单、初版PFMEA、产线规划方案、过程设计初始方案、产品可制造性分析及风险应对报告、初始过程流程图、初始过程特殊特性、初版CP、初版SOP、工艺验证计划、样品包装方案
    - **B样阶段**: 设计变更履历表、更新电芯规格书、更新DFMEA、更新特殊特性清单、制程标准、更新3D数模、更新2D图纸、尺寸链公差计算书、更新BOM清单、更新开模清单、更新三新清单、仿真报告、DV测试报告
    - **C样阶段**: 更新PFMEA、量产产线开发进展报告、更新样品包装方案、更新过程特殊特性清单、更新CP、更新SOP、工艺验证计划、样品历史问题清单、CMK分析报告、CPK分析报告、工程变更履历表、产品可制造性分析与风险应对报告、更新过程流程图、更新过程特殊特性清单、设备停机率统计表&设备故障记录表、工艺验证报告、外观标准书、PV测试报告
    
    ### 📊 检查结果
    
    - **完整**: 所有必要文件都已准备
    - **部分完整**: 大部分文件已准备，少量缺失
    - **不完整**: 多个重要文件缺失
    - **缺失**: 关键文件缺失，需要立即补充

    ### 📁 支持的文件类型

    - 所有文件类型
    """)

def render_history_issues_avoidance_section():
    """Render the history issues avoidance section."""
    st.header("📚 历史问题规避")
    
    st.markdown("""
    历史问题规避功能帮助用户从历史项目中学习，避免重复出现相同的问题。
    
    ### 🎯 功能特点
    
    **📊 问题分析**
    - 历史问题统计分析
    - 问题类型分类
    - 根本原因分析
    
    **🔍 智能匹配**
    - 当前项目与历史项目对比
    - 相似问题自动识别
    - 风险预警提示
    
    **📋 解决方案**
    - 历史解决方案推荐
    - 最佳实践分享
    - 预防措施建议
    
    注意：
    - 当前为预览版，功能仍在完善中；正式版预计10月上线。
    """)

def render_ai_agent_section():
    """Render the AI Agent section."""
    st.header("🤖 AI智能体")

    st.markdown("""
    AI智能体用于通用智能问答与任务协作。你可以就APQP流程、质量工具、或某个具体文档提出问题，AI智能体会自动规划响应，并按需调用工具完成任务。

    ### 功能概述

    - **通用问答**: 关于APQP流程、阶段交付物、控制计划、FMEA等的知识性问题。
    - **文档理解**: 针对上传的文档提出问题（如定位条目、解释字段含义、汇总要点）。
    - **工具调用（自动）**: 按需使用互联网搜索、Python代码编写与执行、数据处理/表格分析、文档解析等能力。
    - **逐步计划**: 自动拆解问题、规划步骤并给出可追溯的中间结论。

    ### 使用流程

    1. 进入“🤖 AI智能体”页面。
    2. 在底部输入问题或任务说明（可引用场景：APQP阶段说明、某Excel表格统计需求、术语解释等）。
    3. AI智能体会自动规划并逐步生成回答，必要时调用工具完成检索、分析或计算。
    4. 如需更深入，可追问或要求生成脚本/表格并查看结果。

    注意：
    - 当前为预览版，工具集合与能力仍在完善中；正式版预计10月上线。
    """)
    
def render_settings_section():
    """Render the settings section."""
    st.header("⚙️ 设置")
    
    st.markdown("""
    本页用于配置系统使用的大语言模型（LLM）后端、模型与推理参数，并查看连接状态与当前配置概览。变更将影响各功能页的AI行为与性能表现。

    ⚠️ 警告：
    - 除非你熟悉大语言模型，否则不要修改设置！
    - Do not change the settings unless you know exactly what you are doing!"

    ### 🔧 后端选择（选择大语言模型）
    - **Ollama (10.31.60.127:11434 / 10.31.60.9:11434)**：使用内网Ollama推理服务。本地/内网模型推理，延迟低、可离线。
    - **OpenAI (sg.uiuiapi.com)**：通过代理网关访问OpenAI兼容接口。功能更全，但依赖网络与API可用性。

    选择后端将决定后续“模型配置”与“连接状态”的内容和可用选项。

    ### 🔗 连接状态
    - 自动检测所选后端的可用性并显示“在线/离线”。
    - Ollama 会检测目标主机连通性；OpenAI 会检测 API Key 与网关连通性。
    - 若显示离线，请先检查网络、代理、服务是否启动。

    ### ⚙️ 模型配置
    视所选后端显示对应的模型与参数。

    #### 当选择 Ollama 时
    - **选择Ollama模型**：从服务端可用模型中选择（如 qwen、llama 等）。
    - **Temperature (温度)**：控制输出随机性。低=更稳定， 高=更有创造性。常用 0.3–0.7。
    - **Top-p (核采样)**：控制词汇多样性，配合温度使用。常用 0.8–0.95。
    - **Top-k**：限制每步可选词汇数量。较小=更保守，默认 40。
    - **Repeat Penalty (重复惩罚)**：抑制赘述/复读，>1 增强惩罚。默认 1.1。
    - **上下文窗口大小 (num_ctx)**：可处理的最大上下文长度（token）。更大可承载更长文档，但更耗内存。
    - **线程数 (num_thread)**：CPU 推理线程数。更高可加快推理（受机器性能限制）。

    #### 当选择 OpenAI 时
    - **选择OpenAI模型**：选择具体模型（例如 gpt-4 兼容模型）。
    - **Temperature (温度)**：同上，0=确定性，2=最大随机性。
    - **Top-p (核采样)**：同上，用于控制多样性。
    - **最大输出长度 (max_tokens)**：限制单次回复的最大token数；更大可能截断更少，但更耗费额度。
    - **Presence Penalty (存在惩罚)**：减少重复主题出现的倾向，>0更鼓励引入新主题。
    - **Frequency Penalty (频率惩罚)**：减少重复词汇出现的倾向，>0更抑制重复用词。
    - **Logit Bias (词汇偏好)**：JSON 配置，微调特定词汇出现概率（高级用法）。

    ### 📋 当前配置概览
    - 实时展示“后端/主机（或API）/当前模型/关键参数”的快照，便于核对是否按预期配置。

    ### 📚 相关文档
    - 快速访问 Ollama 与 OpenAI 的官方文档与参数说明。

    ### 💾 保存与生效
    - 参数调整会自动保存；模型切换会提示“将在下次运行时生效”（避免中断当前任务）。
    - 若遇到异常，可刷新页面或在设置中重新选择后端与模型。

    ### ✅ 建议配置（若不确定）
    - Ollama：Temperature 0.3、Top-p 0.9、Top-k 40、Repeat Penalty 1.1、num_ctx 按默认值、num_thread 依据CPU核心数合理设置。
    - OpenAI：Temperature 0.3、Top-p 1.0、max_tokens 适中、Presence/Frequency Penalty 0.0、Logit Bias 留空。
    """)

def render_faq_section():
    """Render the FAQ section."""
    st.header("❓ 常见问题")
    # Tweak FAQ spacing: reduce gap between question and its bullet answers,
    # and add a bit more space after each Q&A block before the next question.
    st.markdown(
        """
        <style>
        .stMarkdown ol > li { margin: 0.1rem 0 0.9rem 0; }
        .stMarkdown ol > li > p { margin: 0 0 0.2rem 0; }
        .stMarkdown ol > li > ul { margin: 0.2rem 0 0.6rem 0; }
        .stMarkdown ul { margin-top: 0.2rem; }
        .stMarkdown li { line-height: 1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    1. 我对AI的评审结果不满意怎么办？
        - 核对文件是否上传到正确的上传框：例如 A 样文件务必上传到“A样阶段”的上传框，B/C 样同理；“基准文件/待检查文件”也要对应上传到各自区域。
        - 重新运行检查：AI每次会给出不同回应，重跑可得到上次运行漏掉的结果。

    2. 数据会离开公司吗？如何确保数据隐私和安全？
        - 默认在公司内网完成处理，文件保留在公司服务器上，不会外发。
        - 仅在勾选“高性能模式”时，会用阿里云服务器推理；涉密文件不要勾选“高性能模式”。

    3. 在普通模式（即非高性能模式）中运行时，是否有文件外传、数据泄露的风险？
        - 没有风险。所有文件保留在公司服务器上，不会外发；所有分析过程在公司服务器上完成。

    4. 什么是高性能模式？
        - 调用部署在阿里云的大语言模型DeepSeek-V3.1进行推理，速度提升10倍至100倍；
        - 任务运行期间不支持暂停/继续；
        - 涉密文件请勿启用。

    5. 每次运行的最大文件大小和文件数量限制是多少？
        - 无硬性上限，但文件体量越大，运行时间越长；
        - 如果等待太久，建议拆分大批量文件，一次只处理一部分。
        - 也有可能跑得慢是因为多人同时使用。
        - 参考：上传十个20页左右的文件，耗时0.5小时至1小时。如果速度严重偏低，大概率是多人同时使用。

    6. 文件名有要求吗？
        - Windows 非法字符提示：文件名禁止包含以下字符：\\ / : * ? " < > |；同时建议避免结尾空格或句点。

    7. 为什么运行会超时或停滞，如何诊断性能或内存问题？
        - 常见原因：文档体量过大、图片型 PDF、并发段过多或网络波动。
        - 建议：缩小批量、拆分文档、重试，查看“后台日志/流式记录”定位停滞点。
    """)

def render_technical_docs_section(session_id):
    """Render the technical documentation section."""
    st.header("📚 技术文档")
    
    st.markdown("""
    ### 企业标准检查部署细节
    
    #### 服务架构概述
    
    企业标准检查功能依赖以下关键服务：
    1. **MinerU服务** (10.31.60.127:8000) - PDF解析
    2. **Unstructured服务** (10.31.60.11:8000) - Word/PPT解析  
    3. **Ollama服务** (10.31.60.9:11434) - LLM推理
    4. **PQM_AI应用** (10.31.60.127) - 主应用
    
    #### 服务启动与维护
    
    **1. MinerU服务 (10.31.60.127:8000)**
    
    SSH登录服务器：
    ```bash
    ssh -o ServerAliveInterval=130 -o ServerAliveCountMax=6 calb@10.31.60.127
    pwd 0000
    ```
    
    检查服务状态：
    ```bash
    sudo su
    docker ps | grep mineru
    ```
    
    启动服务（如果未运行）：
    ```bash
    # 下载修改后的compose.yaml到MinerU目录
    # 原始文件：wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
    # 已修改版本（取消注释--mem-fraction-static 0.5）：从PQM_AI项目获取
    # 将 [compose.yaml](demonstration/compose.yaml) 复制到服务器MinerU目录
    
    # 进入MinerU目录
    cd /path/to/MinerU
    
    # 使用Docker Compose启动API服务
    docker compose -p mineru -f compose.yaml --profile api up -d
    ```
    
    重启服务：
    ```bash
    docker compose -p mineru -f compose.yaml --profile api restart
    ```
    
    查看日志：
    ```bash
    docker logs mineru-api
    ```
    
    停止服务：
    ```bash
    docker compose -p mineru -f compose.yaml --profile api down
    ```
    
    **Docker镜像信息**：
    - 镜像名：`mineru-sglang:latest`
    - 容器名：`mineru-api`
    - 端口映射：`0.0.0.0:8000->8000/tcp`
    - GPU内存使用：~13.4GB/24GB (RTX 3090)
    
    **重要配置说明**：
    - 使用的compose.yaml已修改：取消注释了`--mem-fraction-static 0.5`参数
    """)

    try:
        compose_path = (Path(__file__).resolve().parent.parent / "demonstration" / "compose.yaml")
        if compose_path.exists():
            with compose_path.open("rb") as f:
                compose_bytes = f.read()
            st.download_button(
                label="⬇️ 下载修改版 compose.yaml",
                data=compose_bytes,
                file_name="compose.yaml",
                mime="application/x-yaml"
                )
        else:
            st.warning(f"未找到compose.yaml：{compose_path}")
    except Exception as e:
        st.warning(f"无法提供compose.yaml下载：{e}")
    st.text("原始文件：https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml")

    st.markdown("""
    - 此修改解决了GPU内存不足导致的OOM（内存溢出）问题
    
    **2. Unstructured服务 (10.31.60.11:8000)**
    
    SSH登录服务器：
    ```bash
    ssh -o ServerAliveInterval=130 -o ServerAliveCountMax=6 file-translation@10.31.60.11
    pwd calb147258
    ```
    
    检查服务状态：
    ```bash
    docker ps | grep unstructured
    ```
    
    启动服务（如果未运行）：
    ```bash
    # 拉取最新镜像
    docker pull downloads.unstructured.io/unstructured-io/unstructured-api:latest
    
    # 启动API容器
    docker run -d --name unstructured-api --restart unless-stopped \\
      -p 8000:8000 \\
      -e PIPELINE_CONCURRENCY_LIMIT=4 \\
      -e ENABLE_CLEANING=1 \\
      downloads.unstructured.io/unstructured-io/unstructured-api:latest
    ```
    
    重启服务：
    ```bash
    docker restart unstructured-api
    ```
    
    查看日志：
    ```bash
    docker logs unstructured-api
    ```
    
    停止服务：
    ```bash
    docker stop unstructured-api
    ```
    
    **3. Ollama服务 (10.31.60.9:11434)**
    
    SSH登录服务器：
    ```bash
    ssh -o ServerAliveInterval=130 -o ServerAliveCountMax=6 translation-service@10.31.60.9
    pwd calb147258
    ```
    
    检查服务状态：
    ```bash
    systemctl status ollama
    ```
    
    启动服务：
    ```bash
    systemctl start ollama
    ```
    
    重启服务：
    ```bash
    systemctl restart ollama
    ```
    
    查看日志：
    ```bash
    journalctl -u ollama -f
    ```
    
    #### 故障诊断与修复
    
    **1. 服务连通性检查**
    
    在PQM_AI应用服务器上执行：
    ```bash
    # 检查MinerU服务
    curl -X GET http://10.31.60.127:8000/health
    
    # 检查Unstructured服务  
    curl -X GET http://10.31.60.11:8000/health
    
    # 检查Ollama服务
    curl -X GET http://10.31.60.9:11434/api/tags
    ```
    
    **2. 常见故障及解决方案**
    
    **MinerU服务故障**：
    - 症状：PDF文件解析失败，返回连接错误
    - 检查：`docker logs mineru-api`
    - 常见原因：GPU内存不足、容器崩溃
    - 解决：重启容器，检查GPU内存使用情况
    
    **Unstructured服务故障**：
    - 症状：Word/PPT文件解析失败
    - 检查：`docker logs unstructured-api`
    - 常见原因：内存不足、端口冲突
    - 解决：重启容器，检查端口8000是否被占用
    
    **Ollama服务故障**：
    - 症状：LLM调用失败，AI功能不可用
    - 检查：`journalctl -u ollama`
    - 常见原因：模型文件损坏、内存不足
    - 解决：重启服务，检查可用模型
    
    **3. 日志位置**
    
    **MinerU日志**：
    ```bash
    # 容器日志
    docker logs mineru-api --tail 100
    
    # 系统日志（如果使用systemd）
    journalctl -u docker --since "1 hour ago"
    ```
    
    **Unstructured日志**：
    ```bash
    # 容器日志
    docker logs unstructured-api --tail 100
    ```
    
    **Ollama日志**：
    ```bash
    # 服务日志
    journalctl -u ollama --since "1 hour ago"
    ```
    
    **PQM_AI应用日志**：
    ```bash
    # Streamlit应用日志（在应用运行终端查看）
    # 或检查系统日志
    journalctl --since "1 hour ago" | grep streamlit
    ```
    
    #### 配置管理
    
    **1. 服务端点配置**
    
    在`config.py`中配置：
    ```python
    CONFIG = {
        "services": {
            "unstructured_api_url": "http://10.31.60.11:8000/general/v0/general",
            "mineru_api_url": "http://10.31.60.127:8000/file_parse"
        },
        "llm": {
            "ollama_host": "http://10.31.60.9:11434"
        }
    }
    ```
    
    **2. 环境变量覆盖**
    
    设置环境变量可覆盖配置文件：
    ```bash
    export UNSTRUCTURED_API_URL="http://10.31.60.11:8000/general/v0/general"
    export OLLAMA_HOST="http://10.31.60.9:11434"
    ```
    
    #### 备份与恢复
    
    **1. 重要文件备份**
    
    ```bash
    # 备份用户数据
    tar -czf pqm_backup_$(date +%Y%m%d).tar.gz \\
      generated_files/ \\
      user_sessions/ \\
      user_settings/ \\
      uploads/
    
    # 备份配置文件
    cp config.py config_backup_$(date +%Y%m%d).py
    ```
    
    **2. 服务配置备份**
    
    ```bash
    # 备份Docker Compose配置
    docker compose -p mineru -f compose.yaml config > mineru_config_backup.yaml
    ```
    
    #### 性能监控
    
    **1. 资源使用监控**
    
    ```bash
    # 检查Docker容器资源使用
    docker stats
    
    # 检查GPU使用情况（MinerU服务器）
    nvidia-smi
    
    # 检查磁盘空间
    df -h
    ```
    
    **2. 服务健康检查脚本**
    
    创建健康检查脚本`health_check.sh`：
    ```bash
    #!/bin/bash
    echo "=== 服务健康检查 ==="
    
    echo "检查MinerU服务..."
    curl -s http://10.31.60.127:8000/health || echo "MinerU服务异常"
    
    echo "检查Unstructured服务..."
    curl -s http://10.31.60.11:8000/health || echo "Unstructured服务异常"
    
    echo "检查Ollama服务..."
    curl -s http://10.31.60.9:11434/api/tags || echo "Ollama服务异常"
    
    echo "=== 检查完成 ==="
    ```
    
    #### 紧急恢复程序
    
    **1. 完全服务重启**
    
    ```bash
    # 在10.31.60.127上
    docker compose -p mineru -f compose.yaml --profile api restart
    
    # 在10.31.60.11上  
    docker restart unstructured-api
    
    # 在10.31.60.9上
    systemctl restart ollama
    
    # 在PQM_AI应用服务器上
    # 重启Streamlit应用
    ```
    
    **2. 服务降级方案**
    
    如果某个服务不可用，可以临时修改配置：
    - 禁用PDF处理：注释掉MinerU相关代码
    - 禁用Word/PPT处理：注释掉Unstructured相关代码
    - 使用备用LLM：修改Ollama配置指向其他服务器
    
    #### 联系信息
    
    - **主要维护人员**：周昭坤
    - **服务器访问**：需要相应服务器的SSH权限
    - **紧急联系**：通过内部通讯工具联系技术团队
    
    #### 相关文档链接
    
    - [MinerU官方文档](https://opendatalab.github.io/MinerU/)
    - [MinerU Docker部署指南](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/)
    - [Unstructured官方文档](https://docs.unstructured.io/)
    - [Ollama官方文档](https://ollama.ai/docs)
    """)



