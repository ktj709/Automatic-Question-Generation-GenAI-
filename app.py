import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import os
import psutil
import gc
import sys
import threading
import time
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import tracemalloc
from dataclasses import dataclass
from collections import deque
from reranker import reranked_retrieve
import pickle
import hashlib
from table_extractor import caption_with_blip
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pandas as pd
from memory_profiler import profile as memory_profile
from pympler import muppy, summary, tracker
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from rouge_score import rouge_scorer
from bert_score import score
from table_extractor import process_pdf_tables
import tempfile

# ------------------- MEMORY CONFIGURATION -------------------
@dataclass
class MemoryConfig:
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    memory_threshold_mb: float = 500.0
    max_history_points: int = 1000
    enable_leak_detection: bool = True
    enable_object_tracking: bool = True
    profile_functions: bool = True

memory_config = MemoryConfig()

@dataclass
class MemorySnapshot:
    timestamp: datetime
    process_memory_mb: float
    system_memory_percent: float
    python_objects_count: int
    gc_collections: Dict[int, int]
    top_objects: List[Dict]
    current_function: Optional[str] = None
    pdf_count: int = 0
    vector_store_size_mb: float = 0.0

@dataclass
class FunctionMemoryProfile:
    function_name: str
    calls_count: int
    total_memory_delta_mb: float
    max_memory_usage_mb: float
    avg_execution_time_ms: float
    memory_leaks_detected: int

class MemoryAnalyzer:
    def __init__(self):
        self.is_monitoring = False
        self.memory_history = deque(maxlen=memory_config.max_history_points)
        self.function_profiles = {}
        self.object_tracker = tracker.SummaryTracker()
        self.process = psutil.Process()
        self.monitoring_thread = None
        self.alert_callbacks = []

        if not tracemalloc.is_tracing():
            tracemalloc.start()

        if 'memory_analyzer' not in st.session_state:
            st.session_state.memory_analyzer = self

    def start_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitoring_thread.start()

    def _monitor_loop(self):
        while self.is_monitoring:
            snapshot = self._capture_memory_snapshot()
            self.memory_history.append(snapshot)
            self._check_memory_alerts(snapshot)
            time.sleep(memory_config.monitoring_interval)

    def _capture_memory_snapshot(self) -> MemorySnapshot:
        memory_info = self.process.memory_info()
        process_memory_mb = memory_info.rss / (1024 * 1024)
        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        all_objects = muppy.get_objects()
        python_objects_count = len(all_objects)
        gc_stats = {i: gc.get_count()[i] for i in range(3)}
        top_objects = self._get_top_objects()
        vector_store_size = self._estimate_vector_store_size()

        return MemorySnapshot(
            timestamp=datetime.now(),
            process_memory_mb=process_memory_mb,
            system_memory_percent=system_memory_percent,
            python_objects_count=python_objects_count,
            gc_collections=gc_stats,
            top_objects=top_objects,
            vector_store_size_mb=vector_store_size
        )

    def _get_top_objects(self, limit=5):
        try:
            all_objects = muppy.get_objects()
            object_summary = summary.summarize(all_objects)
            return [{
                'type': str(row[0]),
                'count': row[1],
                'size_mb': row[2] / (1024 * 1024)
            } for row in object_summary[:limit]]
        except Exception:
            return []

    def _estimate_vector_store_size(self) -> float:
        try:
            if os.path.exists("faiss_index"):
                total_size = 0
                for root, dirs, files in os.walk("faiss_index"):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))
                return total_size / (1024 * 1024)
            return 0.0
        except:
            return 0.0

    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        if snapshot.process_memory_mb > memory_config.memory_threshold_mb:
            for cb in self.alert_callbacks:
                cb({
                    'type': 'HIGH_MEMORY_USAGE',
                    'message': f"Memory usage: {snapshot.process_memory_mb:.1f} MB",
                    'severity': 'WARNING'
                })

    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)

    def get_memory_summary(self) -> Dict:
        if not self.memory_history:
            return {}
        latest = self.memory_history[-1]
        return {
            'current_memory_mb': latest.process_memory_mb,
            'system_memory_percent': latest.system_memory_percent,
            'python_objects': latest.python_objects_count,
            'top_objects': latest.top_objects,
            'vector_store_size_mb': latest.vector_store_size_mb
        }

# ------------------- FUNCTION HELPERS -------------------
def memory_profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()
        result = func(*args, **kwargs)
        after = psutil.Process().memory_info().rss / (1024 * 1024)
        duration = time.time() - start
        st.info(f"⏱️ Function `{func.__name__}` took {duration:.2f}s to execute. Memory used: {after - before:.2f} MB")
        return result
    return wrapper

@st.cache_resource
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def get_llm_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

@memory_profile_function
def extract_text_parallel_with_memory_analysis(pdfs):
    text = ""
    for pdf_file in pdfs:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    return text

@memory_profile_function
def get_text_chunks(text, chunk_size=1500, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

@memory_profile_function
def create_vector_store_with_memory_analysis(text_chunks, file_hash):
    embeddings = get_embeddings_model()
    metadata = [{"chunk_id": i, "file_hash": file_hash} for i in range(len(text_chunks))]
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")
    with open("file_hash.pkl", "wb") as f:
        pickle.dump(file_hash, f)

@memory_profile_function
def smart_retrieve(query, k=4):
    embeddings = get_embeddings_model()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.similarity_search(query, k=k)

@memory_profile_function
def answer_question(question):
    rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    docs = reranked_retrieve(question, k=4, rerank_model_name=rerank_model_name)
    st.write("✅ Reranker applied using:", rerank_model_name)   
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
        Answer the question based only on the following context:

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]

@memory_profile_function
def generate_mcqs_from_context(context, num_questions=5):
    context = context[:4000] + "..."
    prompt = f"""
    Generate {num_questions} multiple-choice questions from the following context. Include 4 options (A to D) and the correct answer.

    Context:
    {context}
    """
    llm = get_llm_model()
    return llm.invoke(prompt)

@memory_profile_function
def extract_images_from_pdf_and_caption(pdfs, llm, max_pages=3):
    captions = []
    for pdf_file in pdfs:
        images = convert_from_bytes(pdf_file.read(), dpi=200, first_page=1, last_page=max_pages)
        for i, img in enumerate(images):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name)
                image_path = tmp.name
            caption = caption_with_blip(img)
            captions.append((img, caption))
    return captions

def compute_file_hash(pdf_files):
    hasher = hashlib.md5()
    for pdf in pdf_files:
        pdf.seek(0)
        while chunk := pdf.read(8192):
            hasher.update(chunk)
        pdf.seek(0)
    return hasher.hexdigest()

def render_memory_dashboard():
    analyzer = st.session_state.memory_analyzer
    st.sidebar.subheader("🧠 Memory Dashboard")
    mem = analyzer.get_memory_summary()
    if mem:
        st.sidebar.metric("Process Memory", f"{mem['current_memory_mb']:.1f} MB")
        st.sidebar.metric("System Memory", f"{mem['system_memory_percent']:.1f}%")
        st.sidebar.metric("Python Objects", f"{mem['python_objects']:,}")
        for obj in mem['top_objects']:
            st.sidebar.text(f"{obj['type']} - {obj['size_mb']:.2f} MB")

# ------------------- MAIN APP -------------------
def main():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    st.set_page_config(page_title="Chat with PDF + Question Generator", layout="wide")
    st.title("📘 Chat with PDF + Question Generator")

    if 'memory_analyzer' not in st.session_state:
        st.session_state.memory_analyzer = MemoryAnalyzer()
        st.session_state.memory_analyzer.add_alert_callback(lambda alert: st.warning(alert['message']))

    render_memory_dashboard()

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdfs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if pdfs:
            file_hash = compute_file_hash(pdfs)
            if st.button("🚀 Process PDFs"):
                with st.spinner("Extracting and indexing..."):
                    text = extract_text_parallel_with_memory_analysis(pdfs)
                    chunks = get_text_chunks(text)
                    create_vector_store_with_memory_analysis(chunks, file_hash)
                    st.success("PDFs processed and indexed!")

            if st.button("📈 Extract Tables from All PDFs"):
                with st.spinner("Analyzing tables..."):
                  summaries = []
                  for pdf in pdfs:
                    st.info(f"🔍 Processing: {pdf.name}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf.read())
                        tmp_path = tmp.name
                    summary = process_pdf_tables(tmp_path, get_llm_model())
                    summaries.append(f"### 📄 {pdf.name}\n\n{summary}")
                  st.session_state["table_summary"] = "\n\n---\n\n".join(summaries)
                  st.success("All tables summarized!")

            if st.button("🖼️ Extract Images & Captions from All PDFs"):
                with st.spinner("Extracting images..."):
                    all_captions = []
                    for pdf in pdfs:
                        st.info(f"🖼️ Extracting from: {pdf.name}")
                        pdf.seek(0)
                        images = convert_from_bytes(pdf.read(), dpi=200, first_page=1, last_page=3)
                        for idx, img in enumerate(images):
                               caption = caption_with_blip(img)
                               all_captions.append((pdf.name, idx + 1, img, caption))
                    st.session_state["image_captions"] = all_captions
                    st.success("All images and captions generated!")

    if "table_summary" in st.session_state:
        st.subheader("📑 Table Summaries")
        st.markdown(st.session_state["table_summary"], unsafe_allow_html=True)
    if "image_captions" in st.session_state:
        st.subheader("🖼️ Image Captions")
        for pdf_name, fig_num, img, caption in st.session_state["image_captions"]:
            st.image(img, caption=f"📄 {pdf_name} — Figure {fig_num}", use_container_width=True)
            st.markdown(f"> {caption}")


    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Answering..."):
            answer = answer_question(question)
            st.success("Answer:")
            st.write(answer)
            st.session_state["last_answer"] = answer

    st.subheader("📊 Evaluate Answer")
    reference = st.text_area("Enter reference answer:")
    if st.button("📟 Evaluate"):
        if "last_answer" in st.session_state and reference.strip():
            try:
                scores = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True).score(reference, st.session_state["last_answer"])
                for k, v in scores.items():
                    st.write(f"{k.upper()}: F1 = {v.fmeasure:.4f}")
                P, R, F1 = score([st.session_state["last_answer"]], [reference], lang='en', verbose=False)
                st.write(f"BERTScore - P: {P[0].item():.4f}, R: {R[0].item():.4f}, F1: {F1[0].item():.4f}")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    st.subheader("🧠 Generate MCQs")
    mcq_count = st.slider("Number of MCQs", 1, 10, 5)
    if st.button("🖋 Generate MCQs"):
        try:
            docs = smart_retrieve("generate questions", k=4)
            context = "\n".join([doc.page_content for doc in docs])
            mcqs = generate_mcqs_from_context(context, mcq_count)
            st.text_area("Generated MCQs", mcqs, height=400)
        except Exception as e:
            st.error(f"Failed to generate MCQs: {e}")

# ------------------- RUN -------------------
if __name__ == "__main__":
    main()