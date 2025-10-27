

import streamlit as st
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from typing import List
from dotenv import load_dotenv
import os

# --- Load API keys & configure env ---
load_dotenv()

st.set_page_config(page_title="PDF QA Reranker", layout="wide")
st.title("🧠 Reranker Module for PDF QA")

# --- Load query input ---
query = st.text_input("Enter your query for reranking:")
rerank_model_name = st.selectbox("Select Reranker Model", [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2"
])
k = st.slider("Top-k reranked chunks to return", 1, 10, 4)

# --- Load embeddings model (LOCAL - NO API NEEDED) ---
@st.cache_resource
def get_embeddings_model():
    """Use local HuggingFace embeddings - no API quota limits"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# --- Reranked Retriever ---
@st.cache_resource
def load_reranker(model_name: str):
    return CrossEncoder(model_name)

# --- Retrieve & Rerank ---
def reranked_retrieve(query: str, k: int = 4, rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List:
    embeddings = get_embeddings_model()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retrieved_docs = db.similarity_search(query, k=10)
    query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
    reranker = load_reranker(rerank_model_name)
    scores = reranker.predict(query_doc_pairs)
    reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    top_k_docs = [doc for doc, _ in reranked[:k]]
    return top_k_docs

# --- UI to trigger reranking ---
if query:
    with st.spinner("🔄 Reranking documents..."):
        try:
            top_docs = reranked_retrieve(query, k=k, rerank_model_name=rerank_model_name)
            st.success(f"Top {k} reranked chunks:")
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")
        except Exception as e:
            st.error(f"Error during reranking: {e}")