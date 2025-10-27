# table_extractor.py
import pdfplumber
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ------------------ PDF Table Extraction & Summarization ------------------

def extract_tables_from_pdf(pdf_path):
    """
    Extracts tables from a PDF using pdfplumber and formats them as Markdown tables.
    """
    markdown_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or not any(table):  # Skip empty tables
                        continue
                    # Format table as Markdown
                    header = table[0]
                    rows = table[1:]
                    md = "| " + " | ".join([str(cell) if cell else "" for cell in header]) + " |\n"
                    md += "| " + " | ".join("---" for _ in header) + " |\n"
                    for row in rows:
                        md += "| " + " | ".join([str(cell) if cell else "" for cell in row]) + " |\n"
                    markdown_tables.append(md)
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []
    
    return markdown_tables

def summarize_table_with_llm(llm, table_markdown: str) -> str:
    """
    Uses an LLM to summarize the contents of a table in Markdown.
    """
    prompt = f"""
You are a helpful assistant that analyzes student score tables.

Given the following table in Markdown format, write a detailed performance summary that:
- Includes average or top-performing students (if identifiable)
- Groups insights by subject and student
- Uses markdown formatting for headings and bullets

Here is the table:

{table_markdown}

Return only the Markdown summary.
"""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        return f"Error generating summary: {e}"

def process_pdf_tables(pdf_path, llm):
    """
    Full pipeline to extract all tables from a PDF and summarize them using a language model.
    """
    tables = extract_tables_from_pdf(pdf_path)
    if not tables:
        return "❌ No tables found in this PDF."
    
    summaries = []
    for i, table in enumerate(tables):
        summary = summarize_table_with_llm(llm, table)
        summaries.append(f"\n\n### 🔹 Table {i+1}\n\n{summary}")
    
    return "\n\n".join(summaries)

# ------------------ BLIP Image Captioning ------------------

# Initialize BLIP components with error handling
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    print(f"Warning: Could not load BLIP model: {e}")
    processor = None
    blip_model = None

def caption_with_blip(image: Image.Image) -> str:
    """Generate a caption for the given image using BLIP."""
    if processor is None or blip_model is None:
        return "BLIP model not available"
    
    try:
        inputs = processor(image, return_tensors="pt")
        output = blip_model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating caption: {e}"