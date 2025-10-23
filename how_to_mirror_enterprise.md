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

## 5. Next Steps for the Special Symbols Tab
- Build a `tabs/special_symbols/workflow.py` module that wraps the new Bisheng flow (see credentials snippet in the user instructions).
- Replace the placeholder UI in `tabs/tab_special_symbols_check.py` with the enterprise-style layout once the workflow surface is ready.
- Wire the new workflow to the dedicated Bisheng backend endpoint created for this tab.

Keep this document updated as further steps are completed.
