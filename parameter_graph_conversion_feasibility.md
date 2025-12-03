# Parameter Check Drawing Conversion Feasibility

## Current handling for "上传图纸文件"
- The parameter check workflow stores uploaded drawing files under the session's `graph` directory and converts them with the same pipeline used for the reference and target files.
- During background execution, PDFs in the drawing directory are parsed via the MinerU-based `process_pdf_folder`, which converts each PDF to a `.txt` by unpacking the first Markdown member from a ZIP response. Other office/text formats follow the same generic conversion helpers.
- This path mirrors the conversion for files uploaded via "上传待检查文件", meaning CAD-style drawings are treated as unstructured documents and may lose layout or panel relationships when fed to the LLM.

## Alternative: `pdf_to_structured_txt.py`
- The standalone `pdf_to_structured_txt.py` script uses PyMuPDF to read words and automatically infers a grid layout (rows × columns) per page based on word coordinates. It groups content into grid cells and writes structured text sections per cell, preserving panel-level order typical of technical drawings.
- Grid parameters can be overridden via CLI flags, and per-page inference limits (max rows/cols) are configurable.

## Feasibility of replacing drawing conversion
- **Technical fit:** The structured extraction is Python-only and operates directly on PDF text positions, so it can be invoked wherever the drawing PDFs are enumerated (e.g., in `process_pdf_folder` when `graph_dir` is processed). It would avoid the MinerU API for drawings and better preserve spatial structure.
- **Dependencies:** Integration would add a hard requirement on `PyMuPDF` (`fitz`). This library is not part of the existing conversion stack and would need to be added to the runtime environment (requirements, packaging, and deployment images).
- **Scope limitation:** The script only handles PDFs. The current pipeline also processes archives and Office/text formats; a drawing-specific branch would still need to route non-PDF uploads through existing converters or reject them explicitly.
- **Performance considerations:** PyMuPDF runs locally; for large, image-heavy drawings, runtime and memory impact should be evaluated compared to the current API-based flow. However, it avoids external API latency and ZIP post-processing.
- **Integration surface:** The graph-file branch in `run_parameters_job` already separates handling of drawing uploads. Replacing the PDF step there with the structured extractor (while keeping other formats on the generic path) would minimize ripple effects. Downstream components consume `.txt` outputs, so the new converter should remain compatible as long as it emits UTF-8 text files.

**Conclusion:** It is feasible to swap the MinerU-based PDF-to-text conversion for drawing uploads with the grid-aware `pdf_to_structured_txt.py` approach, provided PyMuPDF is added as a dependency and non-PDF drawing formats continue to use the existing converters. A targeted replacement in the graph-directory PDF handling would improve layout preservation for technical drawings without disrupting the rest of the pipeline.
