import streamlit as st
from config import CONFIG
from util import resolve_ollama_host
from ollama import Client as OllamaClient
import openai


def render_ai_agent_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("🤖 AI智能体")

    # Ensure the global chat input is visible (other tabs may hide it via CSS)
    st.markdown(
        """
        <style>
        [data-testid=\"stChatInput\"] { display: flex !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Determine backend and initialize clients
    llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama_9')
    if llm_backend in ('ollama_127', 'ollama_9'):
        host = resolve_ollama_host(llm_backend)
        ollama_client = OllamaClient(host=host) 
    else:
        openai.base_url = CONFIG["llm"]["openai_base_url"]
        openai.api_key = CONFIG["llm"]["openai_api_key"]

    # Unified chat history per session
    history_key = f'ai_agent_history_{session_id}'
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    messages = st.session_state[history_key]

    # Render existing history
    for m in messages:
        with st.chat_message(m["role"]):
            st.write(m["content"]) 

    # Chat input pinned at bottom
    prompt = st.chat_input("输入问题或内容，让AI来回复…")
    if not prompt:
        return

    # Show user message immediately and add to history
    with st.chat_message("user"):
        st.write(prompt)
    messages.append({"role": "user", "content": prompt})

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""
        if llm_backend in ('ollama_127','ollama_9'):
            for chunk in ollama_client.chat(
                model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                messages=messages,
                stream=True,
                options={
                    "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                    "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                    "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                    "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                    "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 40001),
                    "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4)
                }
            ):
                new_text = chunk['message']['content']
                response_text += new_text
                response_placeholder.write(response_text)
        else:
            stream = openai.chat.completions.create(
                model=st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"]),
                messages=messages,
                stream=True,
                temperature=st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                top_p=st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                max_tokens=st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                presence_penalty=st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                frequency_penalty=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0)
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                response_text += delta
                response_placeholder.write(response_text)

    # Save assistant response
    messages.append({"role": "assistant", "content": response_text})

