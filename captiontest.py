from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import matplotlib.pyplot as plt

# 1. Create a bar chart and save as image
def create_bar_chart(path):
    plt.figure(figsize=(4, 3))
    subjects = ['Math', 'Physics', 'Chemistry']
    scores = [87, 78, 92]
    plt.bar(subjects, scores, color='skyblue')
    plt.title('Aman Verma - Subject Scores')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# 2. Generate PDF with the image
def generate_test_pdf(pdf_path):
    img_path = "test_chart.png"
    create_bar_chart(img_path)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []

    story.append(Paragraph("Sample Report with Chart", style=None))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Image(img_path, width=5 * inch, height=3 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("This chart shows performance in three science subjects.", style=None))

    doc.build(story)

generate_test_pdf("caption_test.pdf")
print("✅ PDF created as 'caption_test.pdf'")
