# table_extractor.py

from unstructured.partition.pdf import partition_pdf

def extract_tables_from_pdf(pdf_path):
    """
    Extracts table elements from a PDF using unstructured's layout-aware strategy.
    Requires Tesseract to be installed.
    """
    elements = partition_pdf(filename=pdf_path, strategy="hi_res")
    tables = [el for el in elements if el.category == "Table"]
    return tables

def summarize_table_with_llm(llm, table_text: str) -> str:
    """
    Uses an LLM to summarize the contents of a table in Markdown.
    """
    prompt = f"""
You are a helpful assistant that explains and formats tables.

Given the following table content, generate a clean, readable Markdown summary. Format the output using:
- Markdown headings
- Bold names
- Bullet points for subjects and scores
- A closing summary

Here is the table:

{table_text}

Return only the final Markdown.
"""
    return llm.invoke(prompt)

def process_pdf_tables(pdf_path, llm):
    """
    Full pipeline to extract all tables from a PDF and summarize them using a language model.
    """
    tables = extract_tables_from_pdf(pdf_path)
    if not tables:
        return "❌ No tables found in this PDF."

    summaries = []
    for i, table in enumerate(tables):
        summary = summarize_table_with_llm(llm, table.text)
        summaries.append(f"\n\n### 🔹 Table {i+1}\n\n{summary}")

    return "\n\n".join(summaries)

# ------------------ NEW: BLIP Image Captioning ------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_with_blip(image: Image.Image) -> str:
    """Generate a caption for the given image using BLIP."""
    inputs = processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)
