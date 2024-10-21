import os
import shutil
import gradio as gr
from gradio_pdf import PDF
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA



# Global variables
embeddings = None
docsearch = None


def read_file(file_path):
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


def embed_docs(file_path):
    global embeddings, docsearch
    doc_chunks = read_file(file_path)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = FAISS.from_documents(doc_chunks, embeddings)
    return "Document embedded successfully"


def upload_file(file_obj):
    if file_obj is None:
        return None
    UPLOAD_FOLDER = "./uploads"
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, file_obj.name)
    shutil.copy(file_obj.name, file_path)
    return file_path


def extract_info(file_path):
    if not file_path:
        return "No file uploaded", "", "", "", "", ""

    embed_docs(file_path)

    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )

    queries = [
        "What is the employee's name?",
        "What is the employee's salary?",
        "What is the date of the letter?",
        "What is the validity period of the letter?",
        "What is the employee's position or job title?",
        "What is the name of the company issuing the letter?",
    ]

    results = []
    for query in queries:
        output = qa_chain.invoke({"query": query})
        results.append(output["result"])

    return results


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HR Letter Information Extractor")

    with gr.Row():
        pdf_input = PDF(label="Upload HR Letter (PDF)", scale=1, interactive=True)

    extract_btn = gr.Button("Extract Information")

    with gr.Row():
        name_output = gr.Textbox(label="Employee Name")
        salary_output = gr.Textbox(label="Salary")
        date_output = gr.Textbox(label="Letter Date")
        validity_output = gr.Textbox(label="Validity Period")
        position_output = gr.Textbox(label="Position/Job Title")
        company_output = gr.Textbox(label="Company Name")

    extract_btn.click(
        extract_info,
        inputs=[pdf_input],
        outputs=[
            name_output,
            salary_output,
            date_output,
            validity_output,
            position_output,
            company_output,
        ],
    )

# Launch the demo
demo.launch(debug=True)
