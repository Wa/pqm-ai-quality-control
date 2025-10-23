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
- Verified that enterprise session folders live under `enterprise_standard_files/` and that helper utilities prepare them automatically.
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
- Documented outstanding login work: 当前“退出登录”按钮尚未接入逻辑，后续优化时需同步更新本指南。

## 6. Remaining Follow-ups
- Run end-to-end validation with the new Bisheng flow to confirm prompt formats, tweaks, and knowledge-base sync behave as expected for production file sets.
- Decide whether special-symbol post-processing needs tab-specific summaries or can continue reusing the enterprise aggregation helpers.
- Revisit authentication polish (logout wiring, optional password verification) once the workflow stabilises.

Keep this document updated as further steps are completed.
