# How to Mirror the Enterprise Standard Workflow

This document records the steps taken while rebuilding the Special Symbols tab so the same approach can be reused for other tabs.

## 1. Retire the Legacy Special Symbols Implementation
- Removed the old prompt-streaming code from `tabs/tab_special_symbols_check.py`.
- Cleared supporting helpers that were only used by the legacy workflow (for example the special-symbol directory helpers in `util.py`).
- Updated documentation and help content to note the rebuild status.

## 2. Restore Core Shared Utilities
- Reintroduced a sidebar login widget (`util.render_login_widget`) so the app entry point keeps working while the tab is rebuilt.
- Preserved `get_user_session_id` and the shared session/state helpers so other tabs are unaffected.

## 3. Prepare for Enterprise Workflow Mirroring
- Confirmed that the enterprise tab (`tabs/tab_enterprise_standard_check.py`) is the reference for the new design.
- Verified that enterprise session folders live under `uploads/<session_id>/enterprise_standard/` and that helper utilities prepare them automatically.
- Identified that the upcoming rebuild needs:
  - A workflow surface module that encapsulates file handling, backend orchestration, and streaming (mirroring `tabs/enterprise_standard/workflow.py`).
  - Frontend tab code that delegates to the workflow surface instead of issuing direct LLM calls.

## 4. Suggested Implementation Pattern for Future Tabs
1. **Create a workflow surface** (`tabs/<feature>/workflow.py`): expose `prepare_paths`, job submission helpers, and status/result readers.
2. **Refactor the tab UI** (`tabs/tab_<feature>.py`):
   - Initialize session directories via the workflow surface.
   - Use shared file uploader/downloader helpers to keep layout consistent with the enterprise tab.
   - Submit jobs through `backend_client` (or a dedicated client module) and stream results using shared utilities.
3. **Update shared utilities** (`util.py`, `config.py`, etc.) only when the new workflow requires additional shared state.
4. **Document any deviations** inside this guide so future tabs can follow the same pattern with minimal friction.

## 5. Special Symbols Tab Progress
- Added `tabs/special_symbols/workflow.py` and supporting settings so the tab reuses the enterprise surface pattern while targeting the dedicated Bisheng flow and tweaks.
- Replaced the legacy UI in `tabs/tab_special_symbols_check.py` with the enterprise-style layout, including 基准文件 terminology and the demo streaming experience.
- Implemented a background worker (`tabs/special_symbols/background.py`) that mirrors the enterprise pipeline: converts files, syncs the knowledge base, chunk-calls Bisheng with tab-specific tweaks, and aggregates outputs.
- Extended the FastAPI backend to manage both enterprise and special-symbol jobs with the same control signals (start/list/pause/resume/stop) so Streamlit no longer needs to stream prompts directly.
- 登录已恢复为旧版机制：基于 URL 参数的每浏览器自动登录与 per-user JSON 会话管理；设置页“退出登录”现已调用 `deactivate_user_session` 并清除 `auth` 标记。

## 6. Remaining Follow-ups
- ✅ 完成：已在集成环境跑通 Bisheng 流程，确认提示词、KB 同步与结果导出均符合预期。
- ✅ 完成：特殊特性符号暂时沿用企业标准的汇总与导出逻辑，实测满足需求，如后续需要差异化再补充说明。
- ⏳ 待办：如需增强安全性，可按需加入密码校验/单点登录；当前仍为用户名门禁。

Keep this document updated as further steps are completed.
