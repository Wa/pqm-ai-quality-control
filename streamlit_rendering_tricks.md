# Streamlit Rendering Tricks

## Important: Prompt vs Response Rendering

### Problem
When displaying content from `prompt_output.txt` in Streamlit, we discovered that:
- `st.markdown()` shows strikethrough lines on prompt text
- `st.text()` shows clean text without strikethrough lines  
- `st.code()` also shows clean text

### Root Cause
The prompt content contains characters that Markdown interprets as formatting:
- `*` characters - Markdown thinks these are for bold/italic
- `_` characters - Markdown thinks these are for emphasis
- Special characters - Markdown tries to parse them as formatting

When Markdown can't properly parse these characters, it sometimes renders them as **strikethrough lines**.

### Solution
**For Prompts (User Messages):**
```python
# Use st.text() for clean display without Markdown interpretation
prompt_placeholder.text(streamed_prompt.strip())
```

**For LLM Responses (Assistant Messages):**
```python
# Use st.write() for Markdown support in responses
response_placeholder.write(response_text)
```

### Why This Works
- **Prompts**: Usually contain raw text with special characters that shouldn't be interpreted as Markdown
- **Responses**: Often contain formatted text (lists, bold, italic) that benefits from Markdown rendering

### Code Pattern
```python
# Prompt display (plain text)
with st.chat_message("user"):
    prompt_placeholder = st.empty()
    prompt_placeholder.text(streamed_prompt.strip())

# Response display (Markdown)
with st.chat_message("assistant"):
    response_placeholder = st.empty()
    response_placeholder.write(response_text)
```

## Other Streamlit Rendering Notes

### st.write() vs st.text() vs st.markdown()
- `st.write()` - Uses Markdown rendering by default
- `st.text()` - Plain text, no Markdown interpretation
- `st.code()` - Plain text in code block format
- `st.markdown()` - Explicit Markdown rendering

### When to Use Each
- **`st.text()`** - Raw data, prompts, content with special characters
- **`st.write()`** - Formatted content, LLM responses, user-friendly text
- **`st.code()`** - Code snippets, technical data
- **`st.markdown()`** - When you need explicit Markdown control 