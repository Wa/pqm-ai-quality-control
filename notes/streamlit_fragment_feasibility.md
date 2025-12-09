# Streamlit fragments and file-list column feasibility

## Key behaviors of fragments
- `st.fragment` scopes reruns to the decorated block, so user interactions within that block re-execute only the fragment rather than the entire app script.
- Fragments share `st.session_state` with the rest of the app, which allows you to coordinate selections or metadata without forcing global reruns.
- A fragment can be placed anywhere in the layout (including columns or tabs) and invoked as a function, making it straightforward to isolate frequently updated UI pieces.

## Applicability to a file-list column
- Wrapping the right-hand file list in a fragment should prevent delete/rename buttons in that column from triggering reruns of the rest of the page. Only the fragment will refresh, so other tabs or left-hand content remain stable.
- Maintain any underlying file data in `st.session_state` or a backend service so the fragment can read/modify it on each rerun without depending on the outer script execution.
- Ensure expensive operations (e.g., fetching file metadata) are performed inside the fragment or cached; otherwise, external dependencies could still introduce broader reruns.

## Conclusion
Using a fragment for the file list column is feasible and aligns with the fragment design goal of isolating high-churn UI sections. Buttons inside the fragment will refresh only that block while leaving the rest of the app untouched.
