import gradio as gr 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import pdf 
from langchain_community.vectorstores import Chroma 
import ollama
import cv2
import pytesseract
import numpy as np
from datetime import datetime
from PIL import Image
import io
from langchain.schema.retriever import BaseRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from tempfile import NamedTemporaryFile
from langchain.schema import Document
import os
import json


def detect_stamps(image):
    """Detect stamps in the image using contour detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stamp_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 0.8 <= aspect_ratio <= 1.2:  # Looking for roughly square/circular regions
                stamp_regions.append((x, y, w, h))
    
    return stamp_regions

def extract_text_from_image(image):
    """Enhanced text extraction from image using PyTesseract with preprocessing"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create multiple versions of the image for better OCR
    preprocessed_images = []
    
    # Original resized image
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed_images.append(resized)
    
    # Grayscale version
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    preprocessed_images.append(gray)
    
    # Denoised version
    denoised = cv2.fastNlMeansDenoising(gray)
    preprocessed_images.append(denoised)
    
    # Thresholded version
    _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(threshold)
    
    # Configure PyTesseract for better results
    custom_config = r'--oem 3 --psm 6 -l eng+ara'  # Enable both English and Arabic
    
    # Extract text from all versions and combine
    extracted_texts = []
    for img in preprocessed_images:
        text = pytesseract.image_to_string(img, config=custom_config)
        extracted_texts.append(text)
    
    # Combine all extracted texts
    combined_text = ' '.join(extracted_texts)
    return combined_text

def load_retrieve_document(file) -> BaseRetriever:
    extracted_text = ""
    
    if hasattr(file, 'name') and file.name.lower().endswith('.pdf'):
        # Handle PDF
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            with open(file.name, 'rb') as f:    
                tmp_file.write(f.read())
            tmp_path = tmp_file.name
            
        loader = pdf.PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)
        
    else:
        # Handle image file
        image_bytes = file.read() if hasattr(file, 'read') else file
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract text with enhanced method
        extracted_text = extract_text_from_image(image)
        print("Extracted text from image:", extracted_text[:200] + "...")  # Debug print
        documents = [Document(page_content=extracted_text)]

    # Process text with text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased chunk size
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    document_splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(documents=document_splits, embedding=embeddings)
    
    return vector_store.as_retriever(search_kwargs={"k": 2})

def analyze_hr_letter(text: str) -> dict:
    """Enhanced HR letter analysis"""
    print("Analyzing text:", text[:200] + "...")
    
    prompt = """You are an expert HR document analyzer. This is a salary certificate/letter that an employee wants to submit to a fintech company.
    Please analyze the following HR letter carefully and extract these specific details:
    
    1. Full Name: Look for the employee's complete name
    2. Company Name: The current employer of the person
    3. Salary: The current gross salary (including any mentioned allowances or benefits)
    4. Letter Date: When this certificate/letter was issued
    5. Stamps: Whether there are any official stamps on the document
    
    Return ONLY a JSON object with these exact keys:
    {
        "full_name": "extracted name",
        "company_name": "extracted company",
        "salary": "extracted salary with currency",
        "letter_date": "extracted date",
        "stamps_detected": "yes/no"
    }
    
    Important guidelines:
    - Look for Arabic names and translate them if found
    - Include any currency mentioned with the salary
    - The letter date should be in a clear format
    - Look for company letterhead or mentioned employer
    - Return null for any field you cannot find with certainty
    
    Return only the JSON object, no additional text or formatting."""
    
    response = ollama.chat(model='llama3.2', messages=[
        {
            'role': 'system', 
            'content': 'You are an expert HR document analyzer specializing in salary certificates and employment letters.'
        },
        {
            'role': 'user', 
            'content': f"{prompt}\n\nDocument text:\n{text}"
        }
    ])
    
    try:
        content = response['message']['content']
        # Clean up the response more thoroughly
        content = content.replace('```json', '').replace('```', '').strip()
        
        # Ensure the content has proper JSON structure
        if not content.endswith('}'):
            content += '}'
            
        # Remove any trailing commas before closing braces
        content = content.replace(',}', '}')
        
        print("Cleaned response:", content)
        
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw content: {response['message']['content']}")
        # Return a more informative error state
        return {
            "full_name": "Error parsing response",
            "company_name": "Error parsing response",
            "salary": "Error parsing response",
            "letter_date": "Error parsing response",
            "stamps_detected": "Error parsing response"
        }

def format_documents(documents: list) -> str: 
    return "\n\n".join(document.page_content for document in documents)

def rag_chain(file):
    if file is None:
        return None, None, None, None, None
        
    try:
        retriever = load_retrieve_document(file)
        retrieved_documents = retriever.invoke("Extract all relevant information from this document.")
        formatted_context = format_documents(retrieved_documents)
        
        # Add debug print
        print("Retrieved text:", formatted_context[:200] + "...")
        
        # Get HR analysis as structured data
        result = analyze_hr_letter(formatted_context)
        
        # Add debug print
        print("Final result:", result)
        
        return (
            result["full_name"],
            result["company_name"],
            result["salary"],
            result["letter_date"],
            result["stamps_detected"]
        )
    except Exception as e:
        print(f"Error in rag_chain: {e}")  # Add error logging
        return None, None, None, None, None

# Updated interface with multiple outputs
interface = gr.Interface(
    fn=rag_chain,
    inputs=[
        gr.File(
            label="Upload HR Document",
            file_types=[".pdf", ".jpg", ".jpeg", ".png"],
            type="binary"
        )
    ],
    outputs=[
        gr.Textbox(label="Full Name", interactive=False),
        gr.Textbox(label="Company Name", interactive=False),
        gr.Textbox(label="Salary", interactive=False),
        gr.Textbox(label="Letter Date", interactive=False),
        gr.Textbox(label="Stamps Detected", interactive=False)
    ],
    title="HR Letter Analyzer",
    description="Upload an HR document (PDF or image) to automatically extract key information.",
    examples=[
        ["sample.pdf"]
    ]
)
interface.launch(debug=True)

