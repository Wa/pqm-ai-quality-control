import streamlit as st
from pathlib import Path

def render_help_documentation_tab(session_id):
    """Render the help documentation tab."""
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    st.title("📚 帮助文档")
    
    # Create a sidebar for navigation
    with st.sidebar:
        st.header("📖 文档导航")
        
        # Define section mappings
        section_mappings = {
            "概述": "overview",
            "一致性检查": "consistency_check",
            "文件要素检查": "file_elements_check",
            "文件齐套性检查": "file_completeness_check",
            "历史问题规避": "history_issues_avoidance",
            "设置": "settings",
            "常见问题": "faq",
            "技术支持": "support"
        }
        
        # Get the selected section
        selected_section = st.sidebar.selectbox(
            "选择章节",
            list(section_mappings.keys()),
            key=f"help_section_{session_id}"
        )
        
        # Display content based on selection
        if selected_section == "概述":
            render_overview_section()
        elif selected_section == "一致性检查":
            render_consistency_check_section()
        elif selected_section == "文件要素检查":
            render_file_elements_check_section()
        elif selected_section == "文件齐套性检查":
            render_file_completeness_check_section()
        elif selected_section == "历史问题规避":
            render_history_issues_avoidance_section()
        elif selected_section == "设置":
            render_settings_section()
        elif selected_section == "常见问题":
            render_faq_section()
        elif selected_section == "技术支持":
            render_support_section()

def render_overview_section():
    """Render the overview section."""
    st.header("📖 系统概述")
    
    st.markdown("""
    PQM_AI质量控制系统是一个专为汽车行业APQP（Advanced Product Quality Planning）流程设计的智能文档分析工具。
    它利用大语言模型（LLM）技术，帮助质量工程师和项目经理快速、准确地检查和分析APQP相关文档。
    
    ### 🎯 主要功能
    
    **🔍 一致性检查**
    - 控制计划与目标文件的一致性验证
    - 特殊特性符号的自动识别和匹配
    - 智能提示和差异分析
    
    **📋 文件要素检查**
    - APQP文档完整性检查
    - 必要要素的自动识别
    - 规范性验证
    
    **📁 文件齐套性检查**
    - APQP阶段文件完整性检查
    - 缺失文件自动识别
    - 阶段要求对比分析
    
    ### 🚀 技术特点
    
    - **智能分析**: 基于大语言模型的自然语言处理
    - **多格式支持**: 支持Excel、Word、PDF等多种格式
    - **实时处理**: 快速响应，即时反馈
    - **用户友好**: 直观的界面设计，易于操作
    - **多用户支持**: 支持多用户同时使用
    
    ### 🎯 适用场景
    
    - 汽车行业APQP文档检查
    - 质量控制流程优化
    - 项目管理效率提升
    - 合规性验证
    """)

def render_consistency_check_section():
    """Render the consistency check section."""
    st.header("🔍 一致性检查")
    
    st.markdown("""
    一致性检查是系统的核心功能，用于检查控制计划（Control Plan）与目标文件之间的一致性。
    
    ### 📋 功能特点
    
    **🔍 智能识别**
    - 自动识别特殊特性符号（★、☆、/）
    - 智能匹配控制计划与目标文件
    - 支持多种文件格式
    
    **📊 详细分析**
    - 提供详细的一致性分析报告
    - 高亮显示差异和不一致之处
    - 智能建议和优化方案
    
    **⚡ 高效处理**
    - 批量处理多个文件
    - 快速生成分析结果
    - 支持大文件处理
    
    ### 🎯 使用流程
    
    1. **上传文件**: 上传控制计划文件和目标文件
    2. **开始分析**: 点击"开始"按钮启动分析
    3. **查看结果**: 在结果区域查看详细分析
    4. **导出报告**: 可导出分析报告供后续使用
    
    ### 📁 支持的文件类型
    
    - Excel文件 (.xlsx, .xls)
    - Word文档 (.docx, .doc)
    - PDF文件 (.pdf)
    - 文本文件 (.txt)
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
    
    ### 🎯 检查结果
    
    - **通过**: 所有要素检查通过
    - **警告**: 存在轻微问题，需要关注
    - **错误**: 存在严重问题，需要修正
    - **缺失**: 缺少必要要素，需要补充
    """)

def render_file_completeness_check_section():
    """Render the file completeness check section."""
    st.header("📁 文件齐套性检查")
    
    st.markdown("""
    文件齐套性检查确保APQP项目所需的所有文档都已准备就绪，没有遗漏。
    
    ### 📋 检查范围
    
    **📁 立项阶段**
    - 项目立项报告
    - 可行性分析
    - 风险评估
    
    **📁 A样阶段**
    - 产品设计文档
    - 过程设计文档
    - 初始控制计划
    
    **📁 B样阶段**
    - 设计验证报告
    - 过程验证报告
    - 更新控制计划
    
    **📁 C样阶段**
    - 生产准备文档
    - 最终控制计划
    - 交付文档
    
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
    
    ### 📈 数据来源
    
    - 企业内部历史项目数据
    - 行业标准数据库
    - 专家经验库
    - 用户反馈数据
    
    ### 🎯 应用价值
    
    - **降低风险**: 避免重复历史错误
    - **提高效率**: 快速找到解决方案
    - **知识积累**: 持续改进和优化
    - **质量提升**: 基于经验的质量改进
    """)

def render_settings_section():
    """Render the settings section."""
    st.header("⚙️ 设置")
    
    st.markdown("""
    设置页面允许用户配置系统参数和个性化选项。
    
    ### 🔧 主要设置
    
    **🤖 LLM配置**
    - 选择大语言模型
    - 配置模型参数
    - 测试连接状态
    
    **👤 用户设置**
    - 个人信息管理
    - 偏好设置
    - 账户安全
    
    **📁 文件设置**
    - 默认文件路径
    - 文件格式偏好
    - 导出选项
    
    **🔔 通知设置**
    - 邮件通知
    - 系统提醒
    - 报告推送
    
    ### 🎯 设置管理
    
    - **自动保存**: 设置自动保存到本地
    - **云端同步**: 支持设置云端同步
    - **导入导出**: 支持设置备份和恢复
    - **重置功能**: 支持恢复默认设置
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
    
    如果遇到其他问题，请联系技术支持团队。
    """)

def render_support_section():
    """Render the support section."""
    st.header("📞 技术支持")
    
    st.markdown("""
    ### 🎯 支持渠道
    
    **📧 邮件支持**
    - 邮箱: support@example.com
    - 响应时间: 24小时内
    
    **💬 在线客服**
    - 工作时间: 9:00-18:00
    - 实时响应
    
    **📞 电话支持**
    - 热线: 400-123-4567
    - 工作时间: 9:00-18:00
    
    ### 📋 支持内容
    
    - **使用指导**: 系统使用方法和技巧
    - **问题诊断**: 技术问题分析和解决
    - **功能咨询**: 新功能介绍和演示
    - **培训服务**: 用户培训和认证
    
    ### 📚 自助服务
    
    - **帮助文档**: 详细的使用说明
    - **视频教程**: 操作演示视频
    - **常见问题**: 问题解答库
    - **用户论坛**: 用户交流平台
    
    ### 🎯 服务承诺
    
    - **快速响应**: 24小时内响应
    - **专业服务**: 专业技术支持团队
    - **持续改进**: 基于用户反馈持续优化
    - **用户满意**: 以用户满意为目标
    """) 