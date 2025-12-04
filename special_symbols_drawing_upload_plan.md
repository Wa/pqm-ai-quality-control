# Special Symbols Drawing Upload Feasibility and Plan

## Current behavior
- The Special Symbols tab only provisions reference and examined folders via `WorkflowSurface.prepare_paths` and the Streamlit tab uses those two directories for upload/management UI; there is no drawing-specific bucket today. 【F:tabs/tab_special_symbols_check.py†L34-L194】
- The background workflow converts reference and examined files into text, but does not touch any drawing directory because none is defined. 【F:tabs/special_symbols/background.py†L1165-L1233】
- Session directory scaffolding creates `special_reference` and `special_examined` upload folders, but no third folder comparable to the parameters tab’s `graph` directory. 【F:util.py†L277-L293】

## Reference pattern in Parameters tab
- Parameters checks create a dedicated `graph` upload folder alongside `reference` and `target` in both the UI and backend setup. 【F:tabs/tab_parameters_check.py†L74-L96】
- Drawings from the `graph` folder are converted into text with `use_structured_drawings=True`, allowing a specialized extraction path before being merged with the examined texts. 【F:tabs/parameters/background.py†L264-L291】

## Feasibility
- The shared helpers (`ensure_session_dirs`, `WorkflowSurface.prepare_paths`, and the file conversion utilities) are already capable of working with extra upload directories, as demonstrated by the parameters implementation, so adding a drawing bucket for special symbols is structurally consistent.
- The conversion pipeline can reuse the existing drawing-aware PDF handling (`use_structured_drawings`) to preserve structured content if we route drawings through the same examined text directory.
- The background workflow already aggregates examined text from multiple folders; adding one more source does not change downstream consumption because results are driven by text files in `examined_txt_dir`.

## Implementation plan
1) **Provision a drawing upload directory**: Extend `ensure_session_dirs` to create a `special_drawings` folder under `uploads/<session>/special_symbols/drawings` (or similar) and return its path. Plumb this into `render_special_symbols_check_tab` so the tab’s `base_dirs` include the drawing path alongside the generated output root.
2) **Mirror the UI pattern**: Add drawing management in the Special Symbols tab: a third file list/clear button block and a `file_uploader` labeled for drawings, mirroring the parameters tab layout while keeping results separate from the reference/exam lists.
3) **Feed drawings into conversion**: Update `run_special_symbols_job` to ingest the drawing directory after examined files, invoking `process_pdf_folder` with `use_structured_drawings=True` (and the other `process_*` helpers for non-PDFs) so drawings are parsed into `examined_txt_dir` with source annotations disabled.
4) **Maintain downstream flow**: Ensure the knowledge-base sync and streaming logic continue to use the same `examined_txt_dir`, so drawing-derived text is naturally compared without further changes. Add small status messages to differentiate drawing parsing progress, mirroring the parameters job logging.
5) **Edge handling and validation**: Add guardrails for missing drawing dir (graceful warning) and optionally restrict accepted MIME types if drawings are expected to be PDFs/images. Update documentation/help text in the tab to explain the new upload slot.

This mirrors the proven parameters pattern while keeping the special symbols pipeline stable; the change footprint is limited to session directory setup, tab UI, and the conversion pre-processing stage in the background worker.
