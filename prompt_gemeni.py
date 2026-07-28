# pyrefly: ignore [missing-import]
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# Works locally (via .env) AND on Streamlit Cloud (via st.secrets)
HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")

# HuggingFace Inference API (cloud) — Qwen2.5-7B
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=1024,
    temperature=0.5,
)
chat_model = ChatHuggingFace(llm=llm)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["QUESTION", "NUMBER_OF_WORDS", "STYLE"],
    template="""# ROLE

You are an expert academic researcher, technical writer, and domain specialist capable of producing publication-quality content across multiple disciplines. Your writing must demonstrate exceptional conceptual accuracy, logical organization, and linguistic precision.

# INPUTS

Question: {QUESTION}

Word Count: {NUMBER_OF_WORDS}

Writing Style: {STYLE}

# TASK

Generate a comprehensive answer to the given question using exactly the requested writing style and approximately the specified word count.

# WRITING GUIDELINES

1. Answer the question directly with a concise introduction.
2. Present concepts in a logical and coherent progression.
3. Maintain factual accuracy and avoid unsupported claims.
4. Ensure every sentence adds meaningful information.
5. Use smooth transitions to improve readability.
6. Avoid repetition, filler, and unnecessary verbosity.
7. Adapt vocabulary, tone, and sentence complexity according to the selected writing style.
8. If discussing scientific or technical concepts, explain both the theoretical foundation and practical significance where appropriate.
9. End with a concise concluding statement only if it improves completeness.

# STYLE ADAPTATION

Use the value of **{STYLE}** to determine the writing approach.

# OUTPUT REQUIREMENTS

- Target length: Approximately {NUMBER_OF_WORDS} words (±10%).
- Answer only the requested question.
- Do not use headings unless the chosen style explicitly requires them.
- Do not use bullet points unless the chosen style explicitly requires them.
- Do not mention these instructions.
- Produce polished, publication-quality prose ready for direct use.

# OUTPUT

Return only the final answer."""
)

# --- Streamlit UI ---
st.title("📝 Academic AI Writer")
st.caption("Created By Gaurav Rai")

st.divider()

question = st.text_area("❓ Question", placeholder="e.g. What is quantum entanglement?", height=100)

col1, col2 = st.columns(2)
with col1:
    word_count = st.number_input("📏 Word Count", min_value=50, max_value=2000, value=300, step=50)
with col2:
    style = st.selectbox("🎨 Writing Style", [
        "Academic", "Research Paper", "Technical", "Professional",
        "Student Friendly", "Beginner", "Concise", "Detailed",
        "Executive Summary", "Business", "Formal",
        "IEEE Style", "Springer Style", "Elsevier Style", "Nature Style",
        "Interview Answer"
    ])

st.divider()

if st.button("✨ Generate", use_container_width=True):
    if question.strip():
        with st.spinner(f"Generating a {style} answer (~{word_count} words)..."):
            prompt = prompt_template.format(
                QUESTION=question,
                NUMBER_OF_WORDS=word_count,
                STYLE=style
            )
            result = chat_model.invoke(prompt)
            st.subheader("📄 Output")
            st.write(result.content)
    else:
        st.warning("Please enter a question!")