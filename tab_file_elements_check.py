import streamlit as st
from config import CONFIG

def render_file_elements_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    st.title("文件要素审查")
    # Simple two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Simple title and description
        st.title("CALB大模型-智能体平台")
        st.caption("demo：大模型+APQP")
        
        st.header("智能交付物文件审核与协作平台")
        st.write("体验基于大模型的APQP交付物文件审核、协作与管理。安全高效，助力企业智能化升级。")
        
        # Simple buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.link_button("交付物文件审核", "http://10.31.60.127:3001/chat/flow/c2e5134ecffd4d689fb7c2149d010aae")
        with col_btn2:
            st.link_button("镜像网站", "https://bisheng.dataelem.com/chat/flow/ef8d21da9bfc41cf90ad8573f3fe726e")
        
        # Features section
        st.subheader("评审助手功能")
        
        features = [
            ("智能决策支持系统", "TR、APQP预评审，输出AI预评审意见"),
            ("交付物文件审核", "交付物格式核对；文件一致性、齐套性检测；指标达成率检查"),
            ("汇报PPT自动生成", "输出AI预评审报告"),
            ("知识库", "问答系统，快速信息检索")
        ]
        
        for title, description in features:
            with st.expander(title):
                st.write(description)
        
        # Footer
        st.divider()
        st.link_button("平台主页", "http://10.31.60.127:3001")
        st.write("平台账号：**admin** &nbsp; 密码：**Calb@123**")
    
    with col_right:
        # Simple image and video
        st.image(str(CONFIG["files"]["apqp_image"]))
        st.subheader("业务流演示视频")
        st.video(str(CONFIG["files"]["demo_video"])) 