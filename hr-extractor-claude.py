import os
import anthropic
import base64
import httpx
import gradio as gr
from gradio_pdf import PDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
   
client = anthropic.Anthropic()

def encode_image(file_path):
    """Encodes an image file to base64 for inclusion in the prompt."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_info(file):
    """
    Extract information from the uploaded image using Claude and return a structured response.
    """
    if not file:
        return "No file uploaded", "", "", "", "", "", "", "N/A", "N/A"

    # Encode the image into base64 format
    base64_image = encode_image(file.name)
    image_media_type = "image/jpeg" if file.name.lower().endswith('.jpg') or file.name.lower().endswith('.jpeg') else "image/png"

    # Create the message for Claude
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system="You are an expert at analyzing HR and employment letters. Extract the requested information accurately and concisely.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": """
                        Analyze this HR letter image and extract the following information:
                        1. Employee's Name
                        2. Gross Salary (if provided)
                        3. Net Salary (if provided)
                        4. Date of the Letter (in YYYY-MM-DD format)
                        5. Validity Period of the Letter
                        6. Employee's Position or Job Title
                        7. Name of the Company Issuing the Letter
                        8. Is there a stamp on the letter? (Yes/No)
                        9. Potential Fraud Indicators (list any suspicious elements)

                        Provide "Not provided" for missing fields. Format your response as a structured list.
                        """
                    }
                ],
            }
        ],
    )

    # Extract the result content from the response
    result = message.content[0].text

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
        if "Employee's Name:" in line:
            fields["name"] = line.split(": ")[1].strip()
        elif "Gross Salary:" in line:
            fields["gross_salary"] = line.split(": ")[1].strip()
        elif "Net Salary:" in line:
            fields["net_salary"] = line.split(": ")[1].strip()
        elif "Date of the Letter:" in line:
            fields["date"] = line.split(": ")[1].strip()
        elif "Validity Period:" in line:
            fields["validity"] = line.split(": ")[1].strip()
        elif "Position or Job Title:" in line:
            fields["position"] = line.split(": ")[1].strip()
        elif "Name of the Company:" in line:
            fields["company"] = line.split(": ")[1].strip()
        elif "Is there a stamp:" in line:
            fields["stamp"] = line.split(": ")[1].strip()
        elif "Potential Fraud Indicators:" in line:
            fields["fraud"] = line.split(": ")[1].strip()

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

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HR Letter Information Extractor (Claude Version)")
    
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
        image_viewer = gr.Image(label="Image Viewer")

    # Set up event handlers
    file_input.upload(lambda file: file.name, inputs=[file_input], outputs=[image_viewer])

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



