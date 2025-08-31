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
        "🔍 特殊特性符号检查": "special_symbols_check",
        "📊 设计制程检查": "parameters_check",
        "✅ 文件要素检查": "file_elements_check",
        "📁 文件齐套性检查": "file_completeness_check",
        "📋 历史问题规避": "history_issues_avoidance",
        "🤖 AI智能体": "ai_agent",
        "⚙️ 设置": "settings",
        "❓ 常见问题": "faq",
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

        # Current selected section in session_state (default to 🔍 特殊特性符号检查)
        selected_section = st.session_state.get(
            f"help_section_active_{session_id}", "🔍 特殊特性符号检查"
        )
        # Backward compatibility: if an old value without icon is stored, reset to default
        valid_labels = set(section_mappings.keys())
        if selected_section not in valid_labels:
            selected_section = "🔍 特殊特性符号检查"
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
        if selected_section == "🔍 特殊特性符号检查":
            render_special_symbols_check_section()
        elif selected_section == "📊 设计制程检查":
            render_parameters_check_section()
        elif selected_section == "✅ 文件要素检查":
            render_file_elements_check_section()
        elif selected_section == "📁 文件齐套性检查":
            render_file_completeness_check_section()
        elif selected_section == "📋 历史问题规避":
            render_history_issues_avoidance_section()
        elif selected_section == "🤖 AI智能体":
            render_ai_agent_section()
        elif selected_section == "⚙️ 设置":
            render_settings_section()
        elif selected_section == "❓ 常见问题":
            render_faq_section()

def render_special_symbols_check_section():
    """Render the special symbols check section."""
    st.header("🔍 特殊特性符号检查")
    
    st.markdown("""
    特殊特性符号检查功能用于识别和验证控制计划（Control Plan）中的特殊特性符号（★、☆、/），确保其使用符合标准规范。
    
    ### 使用流程

    1. 登录并进入本功能页，确认当前用户会话。
    2. 上传文件：
       - 控制计划文件（必选）
       - 待检查文件（目标文件，必选）
       - 图纸文件（可选）
    3. 运行方式：
       - 若右侧已列出历史文件，请先点击【清空所有文件】按钮，确保以空白环境开始。
       - 按下【开始】：系统会校验已上传文件并清理旧结果，然后直接对“您上传的文件”执行分析流程。
       - 按下【演示】：系统会自动拷贝一套示例文件到您的会话目录，并立即运行同样的分析流程，便于快速上手体验。
    4. 系统自动执行三步分析（用语友好版）：
       - 第一步：AI 助手读取并理解您上传的文档内容，从每个 Excel 工作表中提取与“特殊特性符号（/、★、☆）”相关的信息，形成“阶段性结果”；每个“阶段性结果”对应一个工作表的提取与比对要点，便于逐步核查。
       - 第二步：在阶段性结果的基础上，汇总得到“特殊特性符号不一致结论”，并在页面右侧以流式方式展示，便于您即时查看。
       - 第三步：进一步整理 AI 输出，生成可下载的 Excel 表格（页面会直接展示预览），方便您留档、评审与传阅。
    5. 在页面右侧查看表格与下载的 Excel。如需重跑，可点击【重新开始】。
    
    注意：
    - 本页的 LLM 调用为“无记忆”模式，每次调用相互独立；仅 AI Agent 页保留对话记忆。
    - 若导出的 Excel 中“目标文件/控制计划文件”列为空，通常表示结论文本未包含来源信息。可通过重新生成“结论”步骤（页面已内置强约束提示词）来补齐。

    ### 📁 支持的文件类型

    - 控制计划文件、待检查文件：Excel（.xlsx / .xls）
    - 图纸文件：PDF（.pdf）
    
    如需支持更多格式，请联系：周昭坤。
    """)

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
    
    ### 🎯 检查特点
    
    - **阶段完整性**: 确保每个阶段文件完整
    - **逻辑顺序**: 检查文件间的逻辑关系
    - **版本控制**: 验证文件版本的一致性
    - **依赖关系**: 检查文件间的依赖关系
    
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
    
    st.markdown("""
    ### 🤔 使用问题
    
    **Q: 如何开始使用系统？**
    A: 首先登录系统，然后选择相应的功能模块，上传文件后点击"开始"按钮即可。
    
    **Q: 支持哪些文件格式？**
    A: 系统支持Excel、Word、PDF、文本文件等多种格式。
    
    **Q: 分析结果如何保存？**
    A: 分析结果可以导出为Excel或PDF格式，也可以在线查看。
    
    **Q: 如何处理大文件？**
    A: 系统支持大文件处理，建议分批处理以提高效率。
    
    ### 🔧 技术问题
    
    **Q: 系统响应慢怎么办？**
    A: 可以尝试刷新页面，或者检查网络连接。
    
    **Q: 如何更新模型设置？**
    A: 在设置页面可以修改模型参数，修改后立即生效。
    
    **Q: 支持多用户同时使用吗？**
    A: 是的，系统支持多用户同时使用，每个用户的数据相互隔离。
    
    ### 📞 联系支持
    
    如果遇到其他问题，请联系周昭坤。
    """)

