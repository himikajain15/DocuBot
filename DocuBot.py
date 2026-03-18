import logging
import os
import tempfile

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from streamlit.errors import StreamlitSecretNotFoundError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_groq_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except StreamlitSecretNotFoundError:
        return os.getenv("GROQ_API_KEY", "")


def set_theme():
    st.markdown(
        """
        <style>
            [data-testid="stHeader"] {
                background: transparent !important;
            }
            [data-testid="stToolbar"] {
                right: 1rem !important;
            }
            body, .stApp {
                background:
                    radial-gradient(circle at 18% 22%, rgba(18, 75, 34, 0.42), transparent 26%),
                    radial-gradient(circle at 62% 52%, rgba(82, 110, 38, 0.18), transparent 24%),
                    radial-gradient(circle at 78% 70%, rgba(116, 145, 70, 0.16), transparent 22%),
                    #141414 !important;
                color: #F2F3EA !important;
            }
            .block-container {
                background-color: transparent !important;
                padding-top: 1.8rem !important;
                padding-bottom: 0.5rem !important;
            }
            .stTextInput > div > div > input,
            .stTextArea > div > textarea,
            .stTextInput input {
                background-color: rgba(20, 20, 20, 0.92) !important;
                color: #F2F3EA !important;
                border: 1px solid rgba(160, 194, 89, 0.45) !important;
            }
            .stTextInput label, .stFileUploader label, .stTextArea label {
                color: #F2F3EA !important;
            }
            .stButton > button {
                background: linear-gradient(135deg, #5f8d37, #7ea34b) !important;
                color: #F2F3EA !important;
                border: 1px solid rgba(242, 243, 234, 0.55) !important;
            }
            section[data-testid="stSidebar"] {
                background:
                    radial-gradient(circle at 20% 18%, rgba(18, 75, 34, 0.50), transparent 24%),
                    #141414 !important;
                color: #F2F3EA !important;
                border-right: 1px solid rgba(242, 243, 234, 0.10) !important;
            }
            /* Streamlit file uploader styling (works across multiple Streamlit versions) */
            .stFileUploader,
            div[data-testid="stFileUploader"],
            div[data-testid="stFileUploaderContainer"] {
                width: 100% !important;
                background-color: rgba(20, 20, 20, 0.88) !important;
                color: #F2F3EA !important;
                border: 2px dashed rgba(160, 194, 89, 0.55) !important;
                border-radius: 10px !important;
                padding: 0.8rem !important;
            }
            .stFileUploaderDropzone,
            div[data-testid="stFileUploaderDropzone"] {
                width: 100% !important;
                min-height: 120px !important;
                background-color: rgba(20, 20, 20, 0.88) !important;
                color: #F2F3EA !important;
                border: 2px dashed rgba(160, 194, 89, 0.55) !important;
                border-radius: 10px !important;
                padding: 0.8rem !important;
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                gap: 1rem !important;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
            }
            .stFileUploaderDropzone *,
            div[data-testid="stFileUploaderDropzone"] * {
                color: #F2F3EA !important;
                background: transparent !important;
            }
            div[data-testid="stFileUploaderDropzone"] > div,
            div[data-testid="stFileUploaderDropzone"] > div > div,
            div[data-testid="stFileUploaderDropzone"] > div > div > div {
                background: transparent !important;
            }
            div[data-testid="stFileUploaderDropzone"] button,
            .stFileUploaderDropzone button {
                background-color: rgba(242, 243, 234, 0.08) !important;
                color: #F2F3EA !important;
                border: 1px solid rgba(160, 194, 89, 0.35) !important;
                border-radius: 0.5rem !important;
                padding: 0.4rem 1rem !important;
                min-width: 130px !important;
                box-shadow: none !important;
            }
            div[data-testid="stFileUploaderDropzone"] button:hover,
            .stFileUploaderDropzone button:hover {
                background-color: rgba(242, 243, 234, 0.14) !important;
            }
            div[data-testid="stFileUploaderDropzone"] svg,
            .stFileUploaderDropzone svg {
                fill: #F2F3EA !important;
            }
            /* Force all nested content inside the file uploader to be transparent and dark-themed */
            div[data-testid="stFileUploader"] *,
            div[data-testid="stFileUploaderDropzone"] *,
            .stFileUploader *,
            .stFileUploaderDropzone * {
                background: transparent !important;
                color: #F2F3EA !important;
                border-color: rgba(160, 194, 89, 0.35) !important;
            }
            .stFileUploader label,
            div[data-testid="stFileUploader"] label {
                color: #F2F3EA !important;
            }
            .stMarkdown, p, h1, h2, h3, label, span, div {
                color: #F2F3EA !important;
            }
            div[data-testid="stAlert"] {
                background-color: rgba(20, 20, 20, 0.88) !important;
                color: #F2F3EA !important;
                border: 1px solid rgba(160, 194, 89, 0.35) !important;
            }
            h1 {
                margin-top: 0 !important;
                margin-bottom: 1rem !important;
                font-size: 3.4rem !important;
                line-height: 1.05 !important;
            }
            div[data-testid="stSidebar"] .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0.75rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="DocuBot: Converse with Websites and PDFs", page_icon=":parrot:")
st.title("📚🕵️DocuBot: Converse with Websites and PDFs")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "content_ready" not in st.session_state:
    st.session_state.content_ready = False
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def _summarize_chat_item(item: str, max_chars: int = 80) -> str:
    """Return a short headline-style preview of a chat message."""
    if len(item) <= max_chars:
        return item
    return item[: max_chars - 3].rstrip() + "..."

with st.sidebar:
    st.subheader("Chat History")
    if st.session_state.chat_history:
        for item in st.session_state.chat_history:
            st.write(_summarize_chat_item(item))
    else:
        st.caption("No chat yet.")

set_theme()
groq_api_key = get_groq_api_key()

generic_url = st.text_input("Enter a Website URL")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
current_input = generic_url or (uploaded_file.name if uploaded_file else "")

if current_input != st.session_state.last_input:
    st.session_state.content_ready = False
    st.session_state.last_input = current_input


def fetch_website_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as exc:
        logger.error(f"Website error: {exc}")
        st.error(f"Failed to load website: {exc}")
        return None


def fetch_pdf_docs(uploaded):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        return loader.load()
    except Exception as exc:
        logger.error(f"PDF error: {exc}")
        st.error(f"Failed to read PDF: {exc}")
        return None


if not st.session_state.content_ready:
    if st.button("Analyze"):
        if not groq_api_key.strip():
            st.error("Please add your Groq API key in Streamlit secrets.")
        elif not generic_url and not uploaded_file:
            st.error("Please provide a website URL or upload a PDF.")
        else:
            documents = []

            with st.spinner("Processing..."):
                if generic_url:
                    text = fetch_website_text(generic_url)
                    if text:
                        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = splitter.split_text(text)
                        documents.extend(
                            Document(page_content=chunk, metadata={"source": generic_url})
                            for chunk in chunks
                        )
                        st.success("Loaded content from URL.")
                    else:
                        st.stop()

                if uploaded_file:
                    pdf_docs = fetch_pdf_docs(uploaded_file)
                    if pdf_docs:
                        documents.extend(pdf_docs)
                        st.success("Loaded content from PDF.")
                    else:
                        st.stop()

                if documents:
                    try:
                        embeddings = HuggingFaceEmbeddings()
                        vectorstore = FAISS.from_documents(documents, embeddings)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.llm = ChatGroq(
                            model="Meta-Llama/Llama-4-Scout-17b-16e-Instruct",
                            api_key=groq_api_key,
                        )
                        st.session_state.content_ready = True
                        st.success("Content ready. You can now ask questions.")
                    except Exception as exc:
                        st.error(f"Vector store setup failed: {exc}")
                        st.stop()

if st.session_state.content_ready and st.session_state.vectorstore and st.session_state.llm:
    query = st.text_input("Ask a question based on the document/website:")
    if st.button("Search") and query:
        with st.spinner("Generating answer..."):
            try:
                retrieved_docs = st.session_state.vectorstore.similarity_search(query, k=4)
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                prompt = (
                    "You are answering questions using only the provided context.\n"
                    "If the answer is not present in the context, say you do not know.\n"
                    "Keep the answer concise and clear.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}"
                )
                response = st.session_state.llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                st.session_state.chat_history.append(f"You: {query}")
                st.session_state.chat_history.append(f"Bot: {answer}")
                st.success("Answer:")
                st.write(answer)
            except Exception as exc:
                st.error(f"Failed to generate answer: {exc}")
