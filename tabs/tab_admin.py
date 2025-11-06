import streamlit as st
from pathlib import Path
from backend_client import is_backend_available
import time
from datetime import datetime


def render_admin_tab(session_id):
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("🛡️ 管理员面板")
    
    # Session state for restart tracking
    restart_state_key = "admin_restart_requested"
    restart_timestamp_key = "admin_restart_timestamp"
    
    # Check if restart signal file exists
    restart_file = Path(__file__).parent.parent / ".restart_signal"
    signal_exists = restart_file.exists()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 服务管理")
        
        # Show restart status if restart was recently requested
        if st.session_state.get(restart_state_key):
            restart_time_str = st.session_state.get(restart_timestamp_key, "")
            restart_time_ts = st.session_state.get(restart_timestamp_key + "_ts", time.time())
            if signal_exists:
                st.warning(f"⏳ 重启信号已发送 ({restart_time_str})，等待处理中...")
                st.info("💡 提示：如果信号文件在几秒内消失，说明 start_app.py 已检测到并正在重启服务。")
            else:
                # Signal file was deleted, restart likely happened
                elapsed = time.time() - restart_time_ts
                if elapsed < 30:  # Within 30 seconds
                    st.success(f"✅ 重启信号已被处理 ({restart_time_str})")
                    st.info("💡 服务应该已经重启。如果服务状态显示异常，请检查终端日志。")
                else:
                    # Clear old restart state
                    st.session_state.pop(restart_state_key, None)
                    st.session_state.pop(restart_timestamp_key, None)
                    st.session_state.pop(restart_timestamp_key + "_ts", None)
        
        # Restart services button
        if st.button("🔄 重启服务", key="restart_services", type="primary"):
            try:
                restart_file.touch()
                restart_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                restart_time_ts = time.time()
                st.session_state[restart_state_key] = True
                st.session_state[restart_timestamp_key] = restart_time
                st.session_state[restart_timestamp_key + "_ts"] = restart_time_ts
                st.success(f"✅ 重启信号已发送！ ({restart_time})")
                st.info("⏳ 等待 start_app.py 检测信号并重启服务...")
                st.caption("💡 刷新页面查看最新状态。信号文件存在表示等待处理，消失表示已处理。")
            except Exception as e:
                st.error(f"❌ 创建重启信号失败: {e}")
        
        # Show signal file status
        if signal_exists:
            try:
                file_time = datetime.fromtimestamp(restart_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"📄 信号文件存在 (创建时间: {file_time})")
            except:
                st.caption("📄 信号文件存在")
        else:
            st.caption("📄 信号文件不存在（已处理或未发送）")
        
        st.caption("⚠️ 注意：此操作需要 start_app.py 正在运行才能生效。")
        
        # Service status
        st.markdown("### 📊 服务状态")
        backend_ready = is_backend_available()
        if backend_ready:
            st.success("✅ 后端服务运行中 (http://localhost:8001)")
        else:
            st.error("❌ 后端服务不可用")
            if st.session_state.get(restart_state_key):
                st.warning("⚠️ 如果是重启中，请等待几秒后刷新页面。")
        
        # Frontend status
        try:
            import requests
            response = requests.get("http://localhost:8888", timeout=2)
            if response.status_code == 200:
                st.success("✅ 前端服务运行中 (http://localhost:8888)")
            else:
                st.warning("⚠️ 前端服务响应异常")
        except Exception as e:
            st.error(f"❌ 前端服务不可用: {str(e)}")
            if st.session_state.get(restart_state_key):
                st.warning("⚠️ 如果是重启中，请等待几秒后刷新页面。")
        
        # Auto-refresh if restart was recently requested
        if st.session_state.get(restart_state_key):
            restart_time_ts = st.session_state.get(restart_timestamp_key + "_ts", time.time())
            elapsed = time.time() - restart_time_ts
            if elapsed < 15 and signal_exists:  # Still waiting for processing
                st.caption("⏳ 页面将在 3 秒后自动刷新以检查重启状态...")
                time.sleep(3)
                st.rerun()
            elif elapsed > 60:  # Clear old state after 1 minute
                st.session_state.pop(restart_state_key, None)
                st.session_state.pop(restart_timestamp_key, None)
                st.session_state.pop(restart_timestamp_key + "_ts", None)
    
    with col2:
        st.markdown("### 📝 使用说明")
        st.info("""
        **重启服务功能：**
        
        1. 点击"重启服务"按钮
        2. 系统会创建一个重启信号文件
        3. start_app.py 检测到信号后会自动重启后端和前端服务
        4. 页面会自动刷新
        
        **如何确认重启成功：**
        - 信号文件存在 = 等待处理中
        - 信号文件消失 = 已处理（可能正在重启）
        - 服务状态恢复 = 重启完成
        - 查看终端日志 = 最准确的确认方式
        
        **注意事项：**
        - 确保 `python start_app.py` 正在运行
        - 重启过程中服务会短暂中断（约5-10秒）
        - 如果重启失败，请手动检查终端日志
        - 重启信号文件会立即被删除，不会残留
        """)
        
        # Terminal log hint
        st.markdown("### 🔍 故障排查")
        st.caption("""
        如果重启按钮没有效果：
        
        1. **检查终端**: 运行 `python start_app.py` 的终端应该显示：
           - "🔄 Restart signal detected..."
           - "✅ Backend stopped"
           - "✅ Frontend stopped"
           - "✅ Services restarted successfully!"
        
        2. **检查信号文件**: 
           ```bash
           ls -la .restart_signal
           ```
           如果文件长期存在，说明 start_app.py 没有运行
        
        3. **手动重启**: 如果按钮无效，可以在终端按 Ctrl+C 停止服务，然后重新运行 `python start_app.py`
        """)






