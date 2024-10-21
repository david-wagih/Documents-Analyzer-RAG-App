import os
import shutil
import uuid
import gradio as gr
from gradio_pdf import PDF
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import pytesseract
from PIL import Image
from datetime import datetime, timedelta
from dateutil import parser
import pytesseract






# Global variables
embeddings = None
docsearch = None

def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return read_pdf(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        return read_image(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or image file.")


def read_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text.append(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ",", " "],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    doc_chunks = text_splitter.create_documents(raw_text)
    return doc_chunks


def read_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='ara+eng')
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "،", " "],  # Added Arabic comma
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    doc_chunks = text_splitter.create_documents([text])
    return doc_chunks

def embed_docs(file_path):
    global embeddings, docsearch
    doc_chunks = read_file(file_path)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = FAISS.from_documents(doc_chunks, embeddings)
    return "Document embedded successfully"


def upload_file(file_obj):
    if file_obj is None:
        return None, None
    UPLOAD_FOLDER = "./uploads"
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    
    # Generate a unique filename
    file_extension = os.path.splitext(file_obj.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Copy the file to the new location
    shutil.copy(file_obj.name, file_path)
    return file_path, file_obj.name

def is_letter_valid(letter_date_str):
    try:
        letter_date = parser.parse(letter_date_str, fuzzy=True)
        three_months_ago = datetime.now() - timedelta(days=90)
        return letter_date >= three_months_ago
    except ValueError:
        return "Unable to parse date"


def extract_info(file_path):
    if not file_path:
        return "No file uploaded", "", "", "", "", "", ""

    embed_docs(file_path)

    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )

    queries = [
        "Extract only the employee's name without any additional text.",
        "Extract only the employee's gross salary as a number without any additional text. If not specified, respond with 'Not provided'.",
        "Extract only the employee's net salary as a number without any additional text. If not specified, respond with 'Not provided'.",
        "Extract only the date of the letter in YYYY-MM-DD format without any additional text.",
        "Extract only the validity period of the letter without any additional text.",
        "Extract only the employee's position or job title without any additional text.",
        "Extract only the name of the company issuing the letter without any additional text.",
    ]

    results = []
    for query in queries:
        output = qa_chain.invoke({"query": query})
        results.append(output["result"].strip())

    # Validate the letter date
    letter_date = results[3]
    validity_status = is_letter_valid(letter_date)
    if isinstance(validity_status, bool):
        results[4] = "Valid" if validity_status else "Not valid (older than 3 months)"
    else:
        results[4] = validity_status

    # Handle cases where salary information might not be provided
    gross_salary = results[1] if results[1] != "Not provided" else ""
    net_salary = results[2] if results[2] != "Not provided" else ""

    return results[0], gross_salary, net_salary, results[3], results[4], results[5], results[6]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HR Letter Information Extractor")
    
    with gr.Row():
        file_input = gr.File(label="Upload HR Letter (PDF or Image)")

    with gr.Row():
        pdf_viewer = PDF(label="PDF Viewer", visible=False)
        image_viewer = gr.Image(label="Image Viewer", visible=False)

    extract_btn = gr.Button("Extract Information")

    with gr.Row():
        name_output = gr.Textbox(label="Employee Name")
        gross_salary_output = gr.Textbox(label="Gross Salary")
        net_salary_output = gr.Textbox(label="Net Salary")
        date_output = gr.Textbox(label="Letter Date")
        validity_output = gr.Textbox(label="Validity Period")
        position_output = gr.Textbox(label="Position/Job Title")
        company_output = gr.Textbox(label="Company Name")

    def process_upload(file):
        file_path, file_name = upload_file(file)
        if file_path:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == '.pdf':
                return gr.update(visible=True, value=file_path), gr.update(visible=False)
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                return gr.update(visible=False), gr.update(visible=True, value=file_path)
        return gr.update(visible=False), gr.update(visible=False)

    file_input.upload(
        process_upload,
        inputs=[file_input],
        outputs=[pdf_viewer, image_viewer]
    )

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
        ],
    )

# Launch the demo
demo.launch(debug=True)
