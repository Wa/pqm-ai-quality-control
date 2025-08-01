import streamlit as st
from pathlib import Path

def render_help_documentation_tab(session_id):
    """Render the help documentation tab with sidebar navigation."""
    
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Page header
    st.title("📚 帮助文档")
    st.caption("了解如何使用质量控制系统进行APQP文档分析")
    st.divider()
    
    # Create two columns: sidebar navigation and main content
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 📋 目录")
        
        # Define documentation sections
        sections = {
            "概述": "overview",
            "一致性审查": "consistency_check", 
            "文件要素审查": "file_elements_check",
            "文件齐套性审查": "file_completeness_check",
            "历史问题规避": "history_issues_avoidance",
            "设置": "settings",
            "常见问题": "faq",
            "技术支持": "support"
        }
        
        # Create navigation buttons
        selected_section = st.selectbox(
            "选择文档章节",
            list(sections.keys()),
            key=f"help_nav_{session_id}"
        )
    
    with col2:
        # Render the selected section content
        if selected_section == "概述":
            render_overview_section()
        elif selected_section == "一致性审查":
            render_consistency_check_section()
        elif selected_section == "文件要素审查":
            render_file_elements_check_section()
        elif selected_section == "文件齐套性审查":
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
    st.header("🎯 系统概述")
    
    st.markdown("""
    ### 什么是质量控制系统？
    
    质量控制系统是一个基于人工智能的APQP（Advanced Product Quality Planning，先期产品质量策划）文档分析工具。
    它利用大语言模型（LLM）技术，帮助质量工程师和项目经理快速、准确地审查和分析APQP相关文档。
    
    ### 主要功能
    
    **🔍 一致性审查**
    - 检查控制计划与目标文件的一致性
    - 识别特殊特性符号的不一致
    - 验证参数设置的准确性
    
    **📋 文件要素审查**
    - 检查文档的完整性和规范性
    - 验证必要要素的存在
    - 识别缺失或错误的信息
    
    **📁 文件齐套性审查**
    - 确保所有必要文档都已上传
    - 检查文档版本的匹配性
    - 验证文档结构的完整性
    
    **📈 历史问题规避**
    - 基于历史数据识别潜在问题
    - 提供预防性建议
    - 减少重复性错误
    
    ### 技术架构
    
    - **前端**: Streamlit Web应用
    - **AI引擎**: 支持Ollama（本地）和OpenAI（云端）
    - **数据处理**: Python + Pandas
    - **文件管理**: 本地文件系统
    - **用户管理**: 会话状态管理
    
    ### 适用场景
    
    - 汽车行业APQP文档审查
    - 制造业质量控制
    - 供应商质量管理
    - 新产品开发质量策划
    """)

def render_consistency_check_section():
    """Render the consistency check section."""
    st.header("🔍 一致性审查")
    
    st.markdown("""
    ### 功能说明
    
    一致性审查是系统的核心功能，用于检查控制计划（Control Plan）与目标文件之间的一致性。
    
    ### 工作流程
    
    1. **文件上传**
       - 上传控制计划文件（Excel格式）
       - 上传目标文件（Excel格式）
       - 系统自动解析文件内容
    
    2. **AI分析**
       - 使用大语言模型分析文件内容
       - 识别关键参数和特性
       - 检查一致性差异
    
    3. **结果输出**
       - 生成详细的分析报告
       - 提供不一致项的列表
       - 给出改进建议
    
    ### 特殊特性符号检查
    
    系统特别关注以下特殊特性符号的一致性：
    - **/** (关键特性)
    - **★** (重要特性) 
    - **☆** (一般特性)
    
    ### 输出文件
    
    - `prompt_output.txt`: 详细的分析过程
    - `2_symbol_check_result.txt`: 特殊特性符号检查结果
    
    ### 使用技巧
    
    - 确保上传的文件格式正确
    - 检查文件编码（推荐UTF-8）
    - 验证特殊特性符号的格式
    - 定期保存分析结果
    """)

def render_file_elements_check_section():
    """Render the file elements check section."""
    st.header("📋 文件要素审查")
    
    st.markdown("""
    ### 功能说明
    
    文件要素审查用于检查APQP文档的完整性和规范性，确保所有必要的要素都已包含。
    
    ### 审查要素
    
    **基本信息要素**
    - 产品名称和编号
    - 版本信息
    - 日期和签名
    
    **技术要素**
    - 工艺参数
    - 控制方法
    - 测量设备
    
    **质量要素**
    - 检验标准
    - 不合格品处理
    - 纠正措施
    
    ### 审查标准
    
    系统基于以下标准进行审查：
    - APQP标准要求
    - 行业最佳实践
    - 企业特定要求
    
    ### 输出结果
    
    - 要素完整性评分
    - 缺失要素列表
    - 改进建议
    """)

def render_file_completeness_check_section():
    """Render the file completeness check section."""
    st.header("📁 文件齐套性审查")
    
    st.markdown("""
    ### 功能说明
    
    文件齐套性审查确保APQP项目所需的所有文档都已准备就绪，没有遗漏。
    
    ### 标准文档清单
    
    **第一阶段：计划和确定**
    - 项目计划书
    - 产品设计目标
    - 初始材料清单
    
    **第二阶段：产品设计和开发**
    - 设计FMEA
    - 设计评审记录
    - 工程图纸
    
    **第三阶段：过程设计和开发**
    - 过程流程图
    - 过程FMEA
    - 控制计划
    
    **第四阶段：产品和过程确认**
    - 试生产控制计划
    - 测量系统分析
    - 初始过程能力研究
    
    **第五阶段：反馈、评定和纠正措施**
    - 持续改进计划
    - 客户满意度调查
    - 纠正措施记录
    
    ### 检查方法
    
    1. **文档存在性检查**
    2. **版本匹配性验证**
    3. **内容完整性评估**
    4. **时间序列合理性检查**
    """)

def render_history_issues_avoidance_section():
    """Render the history issues avoidance section."""
    st.header("📈 历史问题规避")
    
    st.markdown("""
    ### 功能说明
    
    历史问题规避功能基于历史数据和经验，帮助识别和预防常见问题。
    
    ### 数据来源
    
    - 历史项目记录
    - 客户反馈数据
    - 质量改进案例
    - 行业最佳实践
    
    ### 问题类型
    
    **设计相关问题**
    - 设计变更频繁
    - 规格不明确
    - 接口定义不清
    
    **过程相关问题**
    - 工艺不稳定
    - 设备能力不足
    - 人员技能欠缺
    
    **质量相关问题**
    - 检验标准不统一
    - 测量系统误差
    - 不合格品处理不当
    
    ### 预防措施
    
    - 早期风险识别
    - 预防性控制措施
    - 经验教训总结
    - 持续改进建议
    """)

def render_settings_section():
    """Render the settings section."""
    st.header("⚙️ 设置")
    
    st.markdown("""
    ### 大语言模型配置
    
    **Ollama（本地）**
    - 支持本地部署的LLM模型
    - 数据安全性高
    - 响应速度快
    - 无需网络连接
    
    **OpenAI（云端）**
    - 使用云端AI服务
    - 模型能力强大
    - 需要网络连接
    - 按使用量计费
    
    ### 参数设置
    
    **Temperature（温度）**
    - 控制输出的随机性
    - 较低值：更确定性
    - 较高值：更创造性
    - 推荐范围：0.1-1.0
    
    **Top-p（核采样）**
    - 控制词汇选择的多样性
    - 范围：0.0-1.0
    - 推荐值：0.9
    
    **Top-k**
    - 限制每次选择的词汇数量
    - 范围：1-100
    - 推荐值：40
    
    **Repeat Penalty（重复惩罚）**
    - 减少重复内容生成
    - 范围：0.0-2.0
    - 推荐值：1.1
    
    ### 设置持久化
    
    - 设置自动保存到本地文件
    - 支持多用户独立配置
    - 重启后设置保持不变
    """)

def render_faq_section():
    """Render the FAQ section."""
    st.header("❓ 常见问题")
    
    st.markdown("""
    ### 文件上传问题
    
    **Q: 支持哪些文件格式？**
    A: 目前支持Excel文件（.xlsx, .xls）格式。建议使用UTF-8编码以确保中文字符正确显示。
    
    **Q: 文件大小有限制吗？**
    A: 建议单个文件不超过10MB，以确保处理速度和稳定性。
    
    **Q: 为什么文件上传失败？**
    A: 请检查文件格式是否正确，文件是否损坏，以及网络连接是否正常。
    
    ### AI分析问题
    
    **Q: 分析结果不准确怎么办？**
    A: 可以尝试调整AI参数设置，或使用不同的模型。确保输入文件格式规范。
    
    **Q: 分析时间太长怎么办？**
    A: 可以尝试使用本地Ollama模型，或检查网络连接速度。
    
    **Q: 如何提高分析质量？**
    A: 确保文件内容清晰、格式规范，使用合适的AI参数设置。
    
    ### 系统使用问题
    
    **Q: 如何保存分析结果？**
    A: 分析结果会自动保存到用户目录下，也可以手动下载。
    
    **Q: 支持多用户同时使用吗？**
    A: 是的，系统支持多用户独立会话，每个用户的设置和结果都是独立的。
    
    **Q: 如何重置设置？**
    A: 可以删除用户设置文件，或使用设置页面的重置功能。
    """)

def render_support_section():
    """Render the support section."""
    st.header("🛠️ 技术支持")
    
    st.markdown("""
    ### 联系支持
    
    **技术问题**
    - 系统功能异常
    - 性能优化建议
    - 新功能需求
    
    **使用指导**
    - 操作流程咨询
    - 最佳实践建议
    - 培训需求
    
    ### 系统要求
    
    **硬件要求**
    - CPU: 4核心以上
    - 内存: 8GB以上
    - 存储: 10GB可用空间
    
    **软件要求**
    - Python 3.8+
    - Streamlit 1.28+
    - 现代浏览器（Chrome, Firefox, Safari, Edge）
    
    **网络要求**
    - 稳定的互联网连接（使用OpenAI时）
    - 本地网络访问（使用Ollama时）
    
    ### 故障排除
    
    **常见错误及解决方案**
    
    1. **连接超时**
       - 检查网络连接
       - 验证服务器地址
       - 尝试重新连接
    
    2. **文件解析错误**
       - 检查文件格式
       - 验证文件编码
       - 重新上传文件
    
    3. **AI服务异常**
       - 检查模型配置
       - 验证API密钥
       - 尝试切换模型
    
    ### 更新日志
    
    **版本 1.0.0**
    - 初始版本发布
    - 支持基本的一致性审查功能
    - 集成Ollama和OpenAI模型
    
    **版本 1.1.0**
    - 添加文件要素审查功能
    - 改进用户界面
    - 优化性能
    
    **版本 1.2.0**
    - 添加文件齐套性审查
    - 实现设置持久化
    - 增强错误处理
    """) 