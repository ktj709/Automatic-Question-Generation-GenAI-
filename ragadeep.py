import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from rouge_score import rouge_scorer
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Local QA Evaluator", layout="wide")
st.title("📄 Local QA Evaluator (No API Required)")

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except:
        st.error("Please install sentence-transformers: pip install sentence-transformers")
        return None

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
        """Calculate how relevant the contexts are to the question"""
        if not contexts:
            return 0.0
        
        relevant_contexts = 0
        for context in contexts:
            # Check if context contains relevant information for the question
            q_similarity = self.semantic_similarity(question, context)
            a_similarity = self.semantic_similarity(answer, context)
            
            # Context is relevant if it's similar to either question or answer
            if q_similarity > 0.3 or a_similarity > 0.3:
                relevant_contexts += 1
        
        return relevant_contexts / len(contexts)
    
    def context_recall(self, reference, contexts):
        """Calculate how well contexts cover the reference answer"""
        if not contexts or not reference:
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        
        # Calculate similarity between reference and combined context
        similarity = self.semantic_similarity(reference, combined_context)
        return similarity
    
    def faithfulness(self, answer, contexts):
        """Calculate if answer is faithful to the contexts"""
        if not contexts or not answer:
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        
        # Calculate similarity between answer and contexts
        similarity = self.semantic_similarity(answer, combined_context)
        return similarity
    
    def answer_correctness(self, answer, reference):
        """Calculate answer correctness using multiple metrics"""
        if not answer or not reference:
            return 0.0
        
        # Semantic similarity
        semantic_score = self.semantic_similarity(answer, reference)
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, answer)
        rouge_score = (rouge_scores['rouge1'].fmeasure + 
                      rouge_scores['rouge2'].fmeasure + 
                      rouge_scores['rougeL'].fmeasure) / 3
        
        # Combine scores
        final_score = (semantic_score * 0.6) + (rouge_score * 0.4)
        return final_score
    
    def evaluate_qa_pair(self, question, answer, reference, contexts):
        """Evaluate a single QA pair"""
        scores = {
            'context_precision': self.context_precision(question, contexts, answer),
            'context_recall': self.context_recall(reference, contexts),
            'faithfulness': self.faithfulness(answer, contexts),
            'answer_correctness': self.answer_correctness(answer, reference)
        }
        return scores

# Initialize evaluator
evaluator = LocalQAEvaluator()

# --- Data Input Section ---
st.header("📥 Upload QA Data")
uploaded_file = st.file_uploader("Upload a JSON or CSV file with question, answer, reference, and source_chunks", type=["json", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    expected_cols = {"question", "answer", "reference", "source_chunks"}
    if not expected_cols.issubset(df.columns):
        st.error(f"Missing required columns. Found: {df.columns.tolist()}")
        st.stop()

    # Convert source_chunks to list if needed
    if isinstance(df.source_chunks.iloc[0], str):
        try:
            df.source_chunks = df.source_chunks.apply(ast.literal_eval)
        except:
            st.error("Error parsing source_chunks. Please ensure they are properly formatted as lists.")
            st.stop()

    # Map to evaluation format
    df["ground_truth"] = df["reference"]
    df["contexts"] = df["source_chunks"]
    df = df.drop(columns=["reference", "source_chunks"])

    data = df.to_dict(orient="records")
    st.success("Data loaded successfully!")
    st.subheader("✅ Sample QA Entry Preview")
    st.json(data[0])

    # --- Local Evaluation ---
    st.header("🏠 Local Evaluation (No API Required)")
    
    if st.button("Run Local Evaluation"):
        try:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, entry in enumerate(data):
                status_text.text(f"Evaluating question {idx + 1}/{len(data)}")
                
                # Evaluate the QA pair
                scores = evaluator.evaluate_qa_pair(
                    entry["question"],
                    entry["answer"],
                    entry["ground_truth"],
                    entry["contexts"]
                )
                
                # Add to results
                result = {
                    "question_id": idx + 1,
                    **scores
                }
                results.append(result)
                
                progress_bar.progress((idx + 1) / len(data))

            status_text.text("Evaluation complete!")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate averages
            avg_scores = results_df.select_dtypes(include=[np.number]).mean()
            
            # Display results
            st.subheader("📊 Local Evaluation Scores")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Context Precision", f"{avg_scores['context_precision']:.3f}")
            with col2:
                st.metric("Context Recall", f"{avg_scores['context_recall']:.3f}")
            with col3:
                st.metric("Faithfulness", f"{avg_scores['faithfulness']:.3f}")
            with col4:
                st.metric("Answer Correctness", f"{avg_scores['answer_correctness']:.3f}")
            
            # Show detailed results
            st.subheader("Detailed Results")
            st.dataframe(results_df)
            
            # Download button
            st.download_button(
                "📥 Download Results", 
                data=results_df.to_csv(index=False), 
                file_name="local_evaluation_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")

    # --- Method Explanation ---
    st.header("🔍 Evaluation Methods")
    with st.expander("How Local Evaluation Works"):
        st.markdown("""
        **Local evaluation uses these methods:**
        
        1. **Context Precision**: Uses semantic similarity to check if contexts are relevant to the question/answer
        2. **Context Recall**: Measures how well contexts cover the reference answer using sentence embeddings
        3. **Faithfulness**: Calculates similarity between answer and contexts to ensure grounding
        4. **Answer Correctness**: Combines semantic similarity and ROUGE scores
        
        **Models Used:**
        - **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic similarity
        - **ROUGE**: For text overlap metrics
        - **TF-IDF**: Fallback for similarity calculations
        
        **Advantages:**
        - No API costs
        - Fast evaluation
        - Works offline
        - Consistent results
        """)

# --- Installation Instructions ---
st.header("📦 Required Dependencies")
with st.expander("Installation Commands"):
    st.code("""
pip install streamlit pandas scikit-learn sentence-transformers nltk rouge-score textstat
    """)

# --- Troubleshooting ---
st.header("🔧 Troubleshooting")
with st.expander("Common Issues"):
    st.markdown("""
    **If you get import errors:**
    ```bash
    pip install sentence-transformers
    pip install rouge-score
    pip install textstat
    ```
    
    **If NLTK download fails:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```
    
    **Memory issues with large datasets:**
    - Process in smaller batches
    - Use simpler similarity methods
    - Reduce context length
    """)