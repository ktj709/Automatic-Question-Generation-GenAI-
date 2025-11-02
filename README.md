## 🧠 Document Intelligence System
## Multi-Modal PDF Question Answering with Reranking, MCQ Generation & Local Evaluation Suite

## 🚀 Overview

This project is an end-to-end Document Intelligence System built using Streamlit, integrating retrieval-augmented generation (RAG), cross-encoder reranking, multimodal extraction, and comprehensive evaluation — all locally executable.

It enables users to:

📄 Upload multi-page PDFs (text, tables, and images)
🤖 Ask questions and get contextually grounded answers
🧩 Generate multiple-choice questions (MCQs) dynamically
🔍 Perform reranked retrieval using a cross-encoder
📊 Evaluate locally with ROUGE, BERTScore, and semantic metrics
🧠 Monitor memory footprint and runtime statistics

## 🧩 Features

## 📝 1. PDF Processing & Chunking

Handles text, tables, and embedded images

Extracts structured information efficiently

Uses intelligent chunking for scalable retrieval

## 🔍 2. Smart Retrieval with Reranker

Employs cross-encoder re-ranking for better relevance

Integrates with FAISS-based vector retrieval

smart_retrieve() ensures precision-driven document chunks

## 💬 3. Question Answering

Uses LLM-backed answer generation

Ensures factual grounding using top reranked context chunks

## 🎯 4. MCQ Generation

Auto-generates MCQs from processed content

Configurable number of questions (e.g., 5, 10, 15)

Suitable for educational and comprehension tasks

## 🧮 5. Evaluation Suite

Compare system answers with reference answers

Compute:

ROUGE-1, ROUGE-2, ROUGE-L

BERTScore

Semantic Similarity (SentenceTransformer)

Context Precision / Recall / Faithfulness / Correctness

Integrated through eval.py and ragadeep.py

## 🧠 6. Memory & Performance Profiling

Tracks:

Memory usage via psutil, pympler, tracemalloc
Processing time per document
Provides runtime statistics for optimization

## 🧰 Tech Stack

| Layer              | Tools / Libraries                                            |
| ------------------ | ------------------------------------------------------------ |
| **Frontend**       | Streamlit                                                    |
| **Core NLP / LLM** | OpenAI / Gemini / HuggingFace Transformers                   |
| **Retrieval**      | LangChain + FAISS                                            |
| **Reranking**      | Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Evaluation**     | ROUGE, BERTScore, SentenceTransformer                        |
| **Visualization**  | Matplotlib, Plotly                                           |
| **PDF Handling**   | PyMuPDF (`fitz`), pdfplumber, ReportLab                      |
| **Performance**    | psutil, pympler, tracemalloc                                 |

## 📁 File Structure

📂 Document-Intelligence-System/
│
├── main.py              # Main Streamlit app (PDF Q&A + MCQ + memory tracking)

├── eval.py              # Evaluation dashboard (ROUGE, BERTScore, semantic)

├── ragadeep.py          # Deep evaluation with readability & interpretability

├── captiontest.py       # PDF image captioning test utility

├── requirements.txt     # All dependencies

└── README.md            # Documentation

## ⚙️ Installation

# Clone the repository
git clone https://github.com/ktj709/Automatic-Question-Generation-GenAI-.git
cd Automatic-Question-Generation-GenAI-

# Install dependencies
pip install -r requirements.txt

## ▶️ Running the Project
🧠 Main Streamlit App

streamlit run main.py

→ Upload PDFs → Ask questions → Generate MCQs → View memory usage

📊 Evaluation App

streamlit run eval.py

→ Compare generated answers vs. references using ROUGE and BERTScore

🧮 Deep Evaluation Suite

streamlit run ragadeep.py

→ Includes advanced metrics like readability, correctness, and coherence

🖼️ Caption Test

python captiontest.py

→ Creates a sample PDF with image captions to verify image extraction





