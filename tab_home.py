import streamlit as st


def render_home_tab(session_id):
    """Render the homepage tab (首页 / home)."""
    # If not logged in, show the standard warning for consistency
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    # Centered section with a soft background color (only affects this tab's content)
    st.markdown(
        """
        <div style="
            width:100%;
            min-height:65vh;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            background-color:#eaf5ff; /* soft calm blue */
            border-radius:12px;
        ">
            <div>
                <div style="font-size:28px;font-weight:600;margin-bottom:8px;">中创新航质量评审助手</div>
                <div style="font-size:20px;color:#4a5568;">CALB Quality Management Bot</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


