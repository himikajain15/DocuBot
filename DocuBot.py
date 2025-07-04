import streamlit as st
import requests
from bs4 import BeautifulSoup
import logging
import tempfile

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(page_title="DocuBot: Converse with Websites & PDFs", page_icon="🦜")
st.title("📚🕵️ DocuBot: Converse with Websites & PDFs")

# Sidebar: API key and theme toggle
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")
    dark_mode = st.toggle("🌙 Dark Mode", value=True)

# Inject CSS for dark/light theme
# Inject custom CSS for dark/light theme
# THEME FUNCTION: Dark theme (full black)
def set_theme(dark=True):
    if dark:
        st.markdown("""
            <style>
                body, .stApp {
                    background-color: #000000 !important;
                    color: #ffffff !important;
                }
                .block-container {
                    background-color: #000000 !important;
                }
                .stTextInput > div > div > input,
                .stTextArea > div > textarea,
                .stTextInput input {
                    background-color: #1a1a1a !important;
                    color: #ffffff !important;
                    border: 1px solid #333333 !important;
                }
                .stTextInput label, .stFileUploader label, .stTextArea label {
                    color: #ffffff !important;
                }
                .stButton > button {
                    background-color: #333333 !important;
                    color: #ffffff !important;
                    border: 1px solid #ffffff !important;
                }
                section[data-testid="stSidebar"] {
                    background-color: #000000 !important;
                    color: #ffffff !important;
                }
                div[data-testid="stFileUploader"] > div > div {
                    background-color: #1a1a1a !important;
                    color: #ffffff !important;
                    border: 2px dashed #555 !important;
                    border-radius: 10px !important;
                    padding: 1rem !important;
                }
                div[data-testid="stFileUploader"] label {
                    color: #ffffff !important;
                }
            </style>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
                body, .stApp {
                    background-color: #bdbbb0 !important;
                    color: #353535 !important;
                }
                .block-container {
                    background-color: #bdbbb0 !important;
                }
                .stTextInput > div > div > input,
                .stTextArea > div > textarea,
                .stTextInput input {
                    background-color: #ffffff !important;
                    color: #353535 !important;
                    border: 1px solid #8a897c !important;
                }
                .stTextInput label, .stFileUploader label, .stTextArea label {
                    color: #353535 !important;
                }
                .stButton > button {
                    background-color: #8a897c !important;
                    color: #ffffff !important;
                    border: 1px solid #bdbbb0 !important;
                }
                section[data-testid="stSidebar"] {
                    background-color: #8a897c !important;
                    color: #ffffff !important;
                }


                div[data-testid="stFileUploader"] > div > div {
                    background-color: #e4d9ff !important;
                    color: #1e2749 !important;
                    border: 2px dashed #1e2749 !important;
                    border-radius: 10px !important;
                    padding: 1rem !important;
                }
                div[data-testid="stFileUploader"] label {
                    color: #353535 !important;
                }
            </style>
        """, unsafe_allow_html=True)

set_theme(dark_mode)

# Session state setup
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'content_ready' not in st.session_state:
    st.session_state.content_ready = False
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

# Inputs
generic_url = st.text_input("Enter a Website URL")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
current_input = (generic_url or uploaded_file.name if uploaded_file else "")

# Detect new input
if current_input != st.session_state.last_input:
    st.session_state.content_ready = False
    st.session_state.last_input = current_input

# Load website content safely
def fetch_website_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        return text
    except Exception as e:
        logger.error(f"Website error: {e}")
        st.error(f"Failed to load website: {e}")
        return None

# Load PDF content safely
def fetch_pdf_docs(uploaded):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"PDF error: {e}")
        st.error(f"Failed to read PDF: {e}")
        return None

# Process on button click
if not st.session_state.content_ready:
    if st.button("Analyze 🌐"):
        if not groq_api_key.strip():
            st.error("Please enter your Groq API Key.")
        elif not generic_url and not uploaded_file:
            st.error("Please provide a website URL or upload a PDF.")
        else:
            documents = []

            with st.spinner("Processing..."):
                # Website
                if generic_url:
                    text = fetch_website_text(generic_url)
                    if text:
                        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = splitter.split_text(text)
                        documents.extend(
                            Document(page_content=chunk, metadata={"source": generic_url})
                            for chunk in chunks
                        )
                        st.success(f"Loaded content from URL.")
                    else:
                        st.stop()

                # PDF
                if uploaded_file:
                    pdf_docs = fetch_pdf_docs(uploaded_file)
                    if pdf_docs:
                        documents.extend(pdf_docs)
                        st.success("Loaded content from PDF.")
                    else:
                        st.stop()

                # Embed and store
                if documents:
                    try:
                        embeddings = HuggingFaceEmbeddings()
                        vectorstore = FAISS.from_documents(documents, embeddings)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.llm = ChatGroq(
                            model="Meta-Llama/Llama-4-Scout-17b-16e-Instruct",
                            api_key=groq_api_key
                        )
                        st.session_state.content_ready = True
                        st.success("✅ Content ready. You can now ask questions.")
                    except Exception as e:
                        st.error(f"Vector store setup failed: {e}")
                        st.stop()

# Ask questions
if st.session_state.content_ready and st.session_state.vectorstore and st.session_state.llm:
    query = st.text_input("Ask a question based on the document/website:")
    if st.button("🔍 Search") and query:
        with st.spinner("Generating answer..."):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                answer = qa_chain.run(query)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
