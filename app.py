import streamlit as st
from pathlib import Path
from rag import KnowledgeAssistant

st.set_page_config(page_title="Enterprise AI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 Enterprise AI Knowledge Assistant")
st.caption("Upload PDF documents, retrieve relevant passages, and ask questions with source references.")

@st.cache_resource
def load_assistant():
    return KnowledgeAssistant()

assistant = load_assistant()

with st.sidebar:
    st.header("⚙️ Settings")
    use_llm = st.toggle("Use Groq LLM for generated answers", value=False)
    st.caption("Without Groq, the app returns a concise answer from retrieved document passages.")
    groq_key = st.text_input("Groq API key", type="password", help="Optional. Get a key from console.groq.com")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Extract PDF text\n2. Split into overlapping chunks\n3. Create semantic embeddings\n4. Retrieve relevant chunks\n5. Generate an answer with citations")

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "ready" not in st.session_state:
    st.session_state.ready = False

uploaded = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)
col1, col2 = st.columns([1, 3])
with col1:
    index_clicked = st.button("📥 Process documents", type="primary", use_container_width=True)
with col2:
    if st.session_state.ready:
        st.success(f"Knowledge base ready: {len(st.session_state.indexed_docs)} PDF(s) indexed.")

if index_clicked:
    if not uploaded:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting text and building semantic index. The embedding model may download the first time..."):
            try:
                docs = [(f.name, f.getvalue()) for f in uploaded]
                count = assistant.ingest_pdfs(docs)
                st.session_state.indexed_docs = [f.name for f in uploaded]
                st.session_state.ready = count > 0
                if count:
                    st.success(f"Indexed {count} text chunks from {len(uploaded)} PDF(s).")
                else:
                    st.error("No readable text was found. If your PDF is scanned, run OCR first.")
            except Exception as exc:
                st.error(f"Could not process PDFs: {exc}")

st.divider()
st.subheader("💬 Ask your documents")
question = st.text_input("Enter a question", placeholder="Example: What are the main objectives described in the report?")
top_k = st.slider("Number of source passages to retrieve", min_value=1, max_value=8, value=4)

if st.button("Ask AI", use_container_width=True):
    if not st.session_state.ready:
        st.warning("Upload and process PDF documents first.")
    elif not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching documents and preparing your answer..."):
            try:
                result = assistant.ask(question, top_k=top_k, use_llm=use_llm, groq_api_key=groq_key)
                st.markdown("### Answer")
                st.write(result["answer"])
                st.markdown("### Document references")
                for i, item in enumerate(result["sources"], 1):
                    with st.expander(f"[{i}] {item['source']} — page {item['page']} (similarity: {item['score']:.3f})"):
                        st.write(item["text"])
            except Exception as exc:
                st.error(f"Question failed: {exc}")

st.caption("Privacy note: PDFs are processed locally by this app. If Groq is enabled, retrieved text is sent to Groq to generate the answer.")
