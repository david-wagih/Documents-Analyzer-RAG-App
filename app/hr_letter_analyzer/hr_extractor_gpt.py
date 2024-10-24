import os
import gradio as gr
from gradio_pdf import PDF
from openai import OpenAI
import base64
from PIL import Image
import io
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


client = OpenAI()

def encode_image(file_path):
    """Encodes an image file to base64."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_document(file_path):
    """Process the document (PDF or image) and return base64 encoded content."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        # Process PDF
        pdf_content = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                pdf_content += page.get_text()
        return pdf_content
    else:
        # Process image
        return resize_and_encode_image(file_path)

def resize_and_encode_image(file_path, max_size=(800, 800)):
    """Resize the image to reduce its size and encode it as base64."""
    with Image.open(file_path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.thumbnail(max_size)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_info(file):
    """Extract information from the uploaded document using GPT-4."""
    if not file:
        return "No file uploaded", "", "", "", "", "", "", "N/A", "N/A"

    document_content = process_document(file.name)
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == '.pdf':
        prompt = f"""Analyze this HR letter PDF content and extract:
1. Employee Name:
2. Gross Salary:
3. Net Salary:
4. Letter Date (YYYY-MM-DD):
5. Validity Period:
6. Job Title:
7. Company Name:
8. Stamp Present? (Yes/No):
9. Fraud Indicators:

Use "Not provided" for missing info. List any fraud indicators.

PDF Content:
{document_content}
"""
    else:
        prompt = f"""Analyze this HR letter image and extract:
1. Employee Name:
2. Gross Salary:
3. Net Salary:
4. Letter Date (YYYY-MM-DD):
5. Validity Period:
6. Job Title:
7. Company Name:
8. Stamp Present? (Yes/No):
9. Fraud Indicators:

Use "Not provided" for missing info. List any fraud indicators.
[data:image/jpeg;base64,{document_content}]
"""

    try:
        if file_extension == '.pdf':
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{document_content}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

        result = response.choices[0].message.content.strip()

        # Parse the response and extract relevant fields
        fields = {
            "name": "Not provided",
            "gross_salary": "Not provided",
            "net_salary": "Not provided",
            "date": "Not provided",
            "validity": "Not provided",
            "position": "Not provided",
            "company": "Not provided",
            "stamp": "Not provided",
            "fraud": "No fraud indicators detected",
        }

        for line in result.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if 'employee name' in key:
                    fields['name'] = value
                elif 'gross salary' in key:
                    fields['gross_salary'] = value
                elif 'net salary' in key:
                    fields['net_salary'] = value
                elif 'letter date' in key:
                    fields['date'] = value
                elif 'validity period' in key:
                    fields['validity'] = value
                elif 'job title' in key:
                    fields['position'] = value
                elif 'company name' in key:
                    fields['company'] = value
                elif 'stamp present' in key:
                    fields['stamp'] = value
                elif 'fraud indicators' in key:
                    fields['fraud'] = value if value else "No fraud indicators detected"

        return (
            fields["name"],
            fields["gross_salary"],
            fields["net_salary"],
            fields["date"],
            fields["validity"],
            fields["position"],
            fields["company"],
            fields["stamp"],
            fields["fraud"],
        )

    except Exception as e:
        print(f"Error processing document: {e}")
        return "Error extracting information", "", "", "", "", "", "", "N/A", "N/A"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HR Letter Information Extractor")
    
    with gr.Row():
        file_input = gr.File(label="Upload HR Letter (PDF or Image)")
        
    extract_btn = gr.Button("Extract Information")

    with gr.Row():
        name_output = gr.Textbox(label="Employee Name")
        gross_salary_output = gr.Textbox(label="Gross Salary")
        net_salary_output = gr.Textbox(label="Net Salary")
        date_output = gr.Textbox(label="Letter Date")
        validity_output = gr.Textbox(label="Validity Period")
        position_output = gr.Textbox(label="Position/Job Title")
        company_output = gr.Textbox(label="Company Name")
        stamp_output = gr.Textbox(label="Stamp Detected")
        fraud_output = gr.Textbox(label="Fraud Status")

    with gr.Row():
        image_viewer = gr.Image(label="Image Viewer", visible=False)
        pdf_viewer = PDF(label="PDF Viewer", visible=False)

    def update_viewer(file):
        if file is None:
            return gr.update(visible=False), gr.update(visible=False)
        
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.pdf':
            return gr.update(visible=False), gr.update(visible=True, value=file.name)
        else:
            return gr.update(visible=True, value=file.name), gr.update(visible=False)

    # Set up event handlers
    file_input.upload(update_viewer, inputs=[file_input], outputs=[image_viewer, pdf_viewer])

    extract_btn.click(
        extract_info,
        inputs=[file_input],
        outputs=[
            name_output,
            gross_salary_output,
            net_salary_output,
            date_output,
            validity_output,
            position_output,
            company_output,
            stamp_output,
            fraud_output,
        ],
    )

# Launch the Gradio demo
demo.launch(debug=True)
