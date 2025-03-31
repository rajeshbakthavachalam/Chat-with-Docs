import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from streamlit_lottie import st_lottie
import json

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Assistant for Document Insights",
    page_icon="🧠",
    layout="wide"
)

# -------------------- Load Local Lottie Animation --------------------
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Replace with your actual path to the animation.json file
lottie_ai_chat = load_lottie_file("animation.json")

# -------------------- Header --------------------
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🧠 AI Assistant for Document Insights")
        st.write(
            "This intelligent assistant uses **FAISS**, **LLaMA**, and **LangChain** to let you "
            "upload a PDF and instantly chat with it. Ideal for extracting insights from technical documents, "
            "manuals, contracts, and more."
        )
    with col2:
        st_lottie(lottie_ai_chat, speed=1, height=220, key="ai-chat")

st.markdown("---")

# -------------------- Helper Functions --------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(path)

def load_local_faiss_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2')
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vector_store_index="faiss_index"):
    vector_store = load_local_faiss_store(vector_store_index)
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama3")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    return RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)

# -------------------- File Upload --------------------
st.subheader("📄 Upload your PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.markdown("✅ File received. Processing...")
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)

    with st.spinner("🔍 Creating vector index..."):
        create_faiss_vector_store(text)

    with st.spinner("⚙️ Initializing chatbot..."):
        qa_chain = build_qa_chain()

    st.success("✅ Ready! Ask anything based on the uploaded document.")
    st.markdown("---")

# -------------------- Question Answering --------------------
if 'qa_chain' in locals():
    st.subheader("💬 Ask a Question About Your Document")
    question = st.text_input("Type your question below:")

    if question:
        with st.spinner("🧠 Generating answer..."):
            answer = qa_chain.run(question)
        st.success("✅ Answer:")
        st.write(answer)
