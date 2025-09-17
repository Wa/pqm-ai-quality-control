import streamlit as st
import streamlit.components.v1 as components
from config import CONFIG

def render_file_elements_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Page subheader
    st.subheader("✅ 文件要素检查")

    # Use Streamlit's official iframe component
    # Prefer internal link first; Bisheng cloud as fallback (reverted)
    primary_url = "http://10.31.60.127:3001/chat/flow/c2e5134ecffd4d689fb7c2149d010aae"
    fallback_url = "https://bisheng.dataelem.com/chat/flow/ef8d21da9bfc41cf90ad8573f3fe726e"

    # Attempt to check reachability AND embeddability (X-Frame-Options / CSP)
    def can_embed(url: str) -> bool:
        try:
            import requests
            resp = requests.head(url, timeout=3, allow_redirects=True)
            if resp.status_code >= 400:
                return False
            headers = {k.lower(): v for k, v in resp.headers.items()}
            # X-Frame-Options policy
            xfo = headers.get("x-frame-options", "").lower()
            if xfo in ("deny", "sameorigin"):
                return False
            # CSP frame-ancestors policy
            csp = headers.get("content-security-policy", "")
            if "frame-ancestors" in csp:
                fa = csp.lower().split("frame-ancestors", 1)[1]
                # If explicitly 'none' or missing any wildcard/localhost, assume blocked
                if "'none'" in fa:
                    return False
                if ("*" not in fa) and ("localhost" not in fa) and ("127.0.0.1" not in fa):
                    return False
            return True
        except Exception:
            return False

    primary_ok = can_embed(primary_url)
    fallback_ok = can_embed(fallback_url)

    # Choose what to display
    url_to_show = primary_url if primary_ok else (fallback_url if fallback_ok else None)

    if url_to_show:
        # Use raw HTML iframe to pass allow attributes suggested by the target docs
        components.html(
            f"""
            <iframe
              src="{url_to_show}"
              style="width: 100%; height: 100%; min-height: 800px; border: 0;"
              frameborder="0"
              allow="fullscreen; clipboard-write; clipboard-read; microphone; camera">
            </iframe>
            """,
            height=820,
            scrolling=True,
        )
    else:
        st.warning("目标站点设置了防嵌入策略（X-Frame-Options/CSP），无法在页面内显示。请使用下方按钮在新标签打开。")

    # Helpful external open links (needed when target site blocks iframes)
    col_a, col_b = st.columns(2)
    with col_a:
        st.link_button("打开主链接(新标签)", primary_url)
    with col_b:
        st.link_button("打开备用链接(新标签)", fallback_url)

    if url_to_show == fallback_url:
        st.caption("已切换到备用链接显示。若嵌入区域仍显示“拒绝连接”，这是目标站点禁止被嵌入所致，请点击上方按钮在新标签页打开。")