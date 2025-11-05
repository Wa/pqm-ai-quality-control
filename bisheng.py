"""
Minimal Streamlit app to call Bisheng History Issues Avoidance flow directly.

- Hardcodes base URL, flow/node IDs, and KB name.
- No backend/session/login/KB upload.
- Simple UI: prompt input, Run button, and response display.

Run with: streamlit run bisheng.py
"""
from __future__ import annotations

import streamlit as st

from typing import Optional

from bisheng_client import (
    find_knowledge_id_by_name,
    call_flow_process,
    parse_flow_answer,
)


# ---- Hardcoded configuration (adjust if needed) ----
BISHENG_BASE_URL: str = "http://10.31.60.11:3001"
BISHENG_API_KEY: Optional[str] = None  # Use None if no key required

# History tab flow settings (mirrors tabs/history/background.py)
FLOW_ID: str = "191af6f3565e415ca9670f1bc2b9117e"
INPUT_NODE_ID: str = "RetrievalQA-f0f31"
MILVUS_NODE_ID: str = "Milvus-cyR5W"
ES_NODE_ID: str = "ElasticKeywordsSearch-1c80e"

# Fixed knowledge base name to use
KB_NAME: str = "rtrtrtkyuiyi_history"


def main() -> None:
    st.set_page_config(page_title="Bisheng Minimal", page_icon="🧩", layout="centered")
    st.title("Bisheng Minimal: History Issues Avoidance")

    st.caption(
        f"Using base_url={BISHENG_BASE_URL}, flow_id={FLOW_ID}, KB={KB_NAME}"
    )

    prompt = st.text_area("Prompt", height=160, placeholder="Type your prompt for Bisheng…")
    run = st.button("Run", type="primary")

    if run:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.status("Resolving knowledge base…", state="running"):
            kb_id = find_knowledge_id_by_name(BISHENG_BASE_URL, BISHENG_API_KEY, KB_NAME)
            if kb_id is None:
                st.error(
                    f"Knowledge base '{KB_NAME}' not found. Create it in Bisheng or adjust KB_NAME."
                )
                return
            st.write(f"KB ID: {kb_id}")

        with st.status("Calling Bisheng flow…", state="running"):
            response = call_flow_process(
                base_url=BISHENG_BASE_URL,
                flow_id=FLOW_ID,
                question=prompt,
                kb_id=kb_id,
                input_node_id=INPUT_NODE_ID,
                api_key=BISHENG_API_KEY,
                session_id=None,
                history_count=0,
                extra_tweaks=None,
                milvus_node_id=MILVUS_NODE_ID,
                es_node_id=ES_NODE_ID,
                timeout_s=120,
                max_retries=1,
                clear_cache=False,
            )

        answer_text, _ = parse_flow_answer(response)

        st.subheader("Response")
        if answer_text.strip():
            st.write(answer_text)
        else:
            st.info("No answer text extracted. Raw response:")
            st.code(str(response), language="json")


if __name__ == "__main__":
    main()

