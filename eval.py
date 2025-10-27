import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings('ignore')

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Streamlit App Config ---
st.set_page_config(page_title="Local QA Evaluator", layout="wide")
st.title("📄 Local QA Evaluator (No API Required)")

# --- Load Sentence Transformer ---
@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except:
        st.error("Please install sentence-transformers: pip install sentence-transformers")
        return None

# --- Evaluator Class ---
class LocalQAEvaluator:
    def __init__(self):
        self.sentence_model = load_sentence_transformer()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.stop_words = set(stopwords.words('english'))
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence transformers"""
        if not self.sentence_model:
            return self.tfidf_similarity(text1, text2)
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def tfidf_similarity(self, text1, text2):
        """Fallback TF-IDF similarity"""
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    
    def context_precision(self, question, contexts, answer):
        """Measure how relevant contexts are to the question/answer"""
        if not contexts:
            return 0.0
        relevant_contexts = 0
        for context in contexts:
            q_sim = self.semantic_similarity(question, context)
            a_sim = self.semantic_similarity(answer, context)
            if q_sim > 0.3 or a_sim > 0.3:
                relevant_contexts += 1
        return relevant_contexts / len(contexts)
    
    def context_recall(self, reference, contexts):
        """Measure how well contexts cover the reference answer"""
        if not contexts or not reference:
            return 0.0
        combined_context = " ".join(contexts)
        similarity = self.semantic_similarity(reference, combined_context)
        return similarity
    
    def faithfulness(self, answer, contexts):
        """Check if answer is grounded in contexts"""
        if not contexts or not answer:
            return 0.0
        combined_context = " ".join(contexts)
        similarity = self.semantic_similarity(answer, combined_context)
        return similarity
    
    def answer_correctness(self, answer, reference):
        """Calculate answer correctness using semantic + ROUGE"""
        if not answer or not reference:
            return 0.0
        semantic_score = self.semantic_similarity(answer, reference)
        rouge_scores = self.rouge_scorer.score(reference, answer)
        rouge_score = (rouge_scores['rouge1'].fmeasure +
                      rouge_scores['rouge2'].fmeasure +
                      rouge_scores['rougeL'].fmeasure) / 3
        final_score = (semantic_score * 0.6) + (rouge_score * 0.4)
        return final_score
    
    def evaluate_qa_pair(self, question, answer, reference, contexts):
        """Evaluate a single QA pair"""
        return {
            'context_precision': self.context_precision(question, contexts, answer),
            'context_recall': self.context_recall(reference, contexts),
            'faithfulness': self.faithfulness(answer, contexts),
            'answer_correctness': self.answer_correctness(answer, reference)
        }

# --- Initialize Evaluator ---
evaluator = LocalQAEvaluator()

# --- Input Section ---
st.header("📥 Input QA Data")

input_mode = st.radio("Choose Input Mode:", ["Upload File", "Manual Entry"])
data = None

# ===== FILE UPLOAD MODE =====
if input_mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a JSON or CSV file with question, answer, reference, and source_chunks",
        type=["json", "csv"]
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        expected_cols = {"question", "answer", "reference", "source_chunks"}
        if not expected_cols.issubset(df.columns):
            st.error(f"Missing required columns. Found: {df.columns.tolist()}")
            st.stop()

        if isinstance(df.source_chunks.iloc[0], str):
            try:
                df.source_chunks = df.source_chunks.apply(ast.literal_eval)
            except:
                st.error("Error parsing source_chunks. Ensure they’re valid Python lists.")
                st.stop()

        df["ground_truth"] = df["reference"]
        df["contexts"] = df["source_chunks"]
        df = df.drop(columns=["reference", "source_chunks"])

        data = df.to_dict(orient="records")
        st.success("✅ Data loaded successfully!")
        st.subheader("🔍 Sample QA Entry")
        st.json(data[0])

# ===== MANUAL ENTRY MODE =====
elif input_mode == "Manual Entry":
    st.subheader("📝 Enter One QA Pair")

    question = st.text_area("Question", placeholder="Enter your question here...")
    answer = st.text_area("Generated Answer", placeholder="Enter the model’s answer...")
    reference = st.text_area("Reference Answer (Ground Truth)", placeholder="Enter the correct answer...")
    contexts_raw = st.text_area("Source Contexts (comma-separated)", placeholder="context1, context2, ...")

    if question and answer and reference and contexts_raw:
        contexts = [c.strip() for c in contexts_raw.split(",") if c.strip()]
        data = [{
            "question": question,
            "answer": answer,
            "ground_truth": reference,
            "contexts": contexts
        }]

# --- Evaluation Section ---
if data:
    st.header("🏠 Local Evaluation (No API Required)")
    if st.button("Run Local Evaluation"):
        try:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, entry in enumerate(data):
                status_text.text(f"Evaluating question {idx + 1}/{len(data)}")
                scores = evaluator.evaluate_qa_pair(
                    entry["question"],
                    entry["answer"],
                    entry["ground_truth"],
                    entry["contexts"]
                )
                results.append({"question_id": idx + 1, **scores})
                progress_bar.progress((idx + 1) / len(data))

            status_text.text("✅ Evaluation complete!")
            results_df = pd.DataFrame(results)
            avg_scores = results_df.select_dtypes(include=[np.number]).mean()

            st.subheader("📊 Average Scores")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Context Precision", f"{avg_scores['context_precision']:.3f}")
            with col2: st.metric("Context Recall", f"{avg_scores['context_recall']:.3f}")
            with col3: st.metric("Faithfulness", f"{avg_scores['faithfulness']:.3f}")
            with col4: st.metric("Answer Correctness", f"{avg_scores['answer_correctness']:.3f}")

            st.subheader("Detailed Results")
            st.dataframe(results_df)

            st.download_button(
                "📥 Download Results",
                data=results_df.to_csv(index=False),
                file_name="local_evaluation_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")

# --- Info Sections ---
st.header("🔍 Evaluation Methods")
with st.expander("How Local Evaluation Works"):
    st.markdown("""
    **Local evaluation uses:**
    1. **Context Precision:** Measures if contexts are relevant to question/answer.
    2. **Context Recall:** Checks how well contexts cover the reference answer.
    3. **Faithfulness:** Verifies if answer stays grounded in provided contexts.
    4. **Answer Correctness:** Combines semantic + ROUGE overlap.

    **Models Used**
    - Sentence Transformer: `all-MiniLM-L6-v2`
    - ROUGE for text overlap
    - TF-IDF fallback

    **Benefits**
    - No API or cloud dependency  
    - Offline and reproducible  
    - Lightweight and interpretable
    """)

st.header("📦 Installation")
with st.expander("Dependencies"):
    st.code("""
pip install streamlit pandas scikit-learn sentence-transformers nltk rouge-score textstat
    """)

st.header("🔧 Troubleshooting")
with st.expander("Common Issues"):
    st.markdown("""
    **If you get import errors:**
    ```bash
    pip install sentence-transformers
    pip install rouge-score
    pip install textstat
    ```

    **If NLTK data missing:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

    **For memory issues:**
    - Evaluate smaller batches  
    - Reduce context lengths  
    - Use TF-IDF fallback
    """)
