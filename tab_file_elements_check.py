import streamlit as st
from config import CONFIG

def render_file_elements_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Page subheader
    st.subheader("✅ 文件要素检查")

    # Try native components first (no built-in webview; use iframe)
    primary_url = "http://10.31.60.127:3001/chat/flow/c2e5134ecffd4d689fb7c2149d010aae"
    fallback_url = "https://bisheng.dataelem.com/chat/flow/ef8d21da9bfc41cf90ad8573f3fe726e"

    # Attempt to check reachability of the primary URL server-side
    try:
        import requests
        resp = requests.get(primary_url, timeout=3)
        use_primary = resp.status_code == 200
    except Exception:
        use_primary = False

    url_to_show = primary_url if use_primary else fallback_url

    st.markdown(
        f"""
        <iframe src="{url_to_show}" style="width:100%; height:80vh; border:none;" allow="clipboard-read; clipboard-write; microphone; camera"></iframe>
        """,
        unsafe_allow_html=True,
    )

    if not use_primary:
        st.caption("已切换到备用链接显示。")