from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from typing import List, Optional
import uuid
import os
from dotenv import load_dotenv
import ssl
import certifi
import chromadb
import warnings
from urllib3.exceptions import InsecureRequestWarning
from config import settings
from watchfiles import awatch
import asyncio
from logger_config import setup_logger
import logging
import uvicorn
from datetime import datetime, timedelta
import re
from typing import Dict, Any
import json

logger = setup_logger()

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL verification warnings
warnings.simplefilter('ignore', InsecureRequestWarning)

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma

# Document processing imports
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from pypdf import PdfReader
from PIL import Image
import io
import graphviz

# Load environment variables from the backend/.env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Validate environment variables
settings.validate()

# Convert API key to SecretStr
api_key = SecretStr(settings.OPENAI_API_KEY)

# Initialize OpenAI with the API key
embeddings = OpenAIEmbeddings(
    api_key=api_key
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["Content-Length", "Content-Range"]
)

# Initialize vector store with the new client architecture
CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Create Chroma client with new architecture
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Get or create collection
collection_name = "documents"
try:
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    print(f"Error creating collection: {e}")
    # Try to get existing collection
    collection = chroma_client.get_collection(name=collection_name)

# Create langchain wrapper for the collection
vector_store = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)

class ChatRequest(BaseModel):
    message: str
    documentIds: List[str]

class HRLetterAnalysis(BaseModel):
    employee_name: str | None
    salary: float | None
    letter_date: str | None
    is_valid: bool
    has_stamp: bool
    validity_reason: str
    confidence_score: float

def format_hr_letter_analysis(text: str, analysis_data: dict) -> str:
    """Format the HR letter analysis in a clean markdown structure."""
    
    markdown_output = f"""
## 📄 HR Letter Analysis

### 👤 Employee Information
- **Name**: {analysis_data.get('employee_name', 'Not specified')}
- **Employee ID**: {analysis_data.get('employee_id', 'Not specified')}
- **Department**: {analysis_data.get('department', 'Not specified')}
- **Position**: {analysis_data.get('position', 'Not specified')}
- **Date of Joining**: {analysis_data.get('joining_date', 'Not specified')}

### 💰 Salary Details
- **Gross Salary**: {analysis_data.get('gross_salary', 'Not specified')} EGP/month
- **Net Salary**: {analysis_data.get('net_salary', 'Not specified')} EGP/month

### 📅 Letter Details
- **Date**: {analysis_data.get('letter_date', 'Not specified')}
- **Validity**: {'✅ Valid' if analysis_data.get('is_valid') else '❌ Invalid'}
- **Reason**: {analysis_data.get('validity_reason', 'Not specified')}
- **Has Official Stamp**: {'✅ Yes' if analysis_data.get('has_stamp') else '❌ No'}

### 🔍 Verification Status
- **Confidence Score**: {analysis_data.get('confidence_score', 0.95) * 100}%
- **Last Verified**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*Note: This is an automated analysis of the HR letter. Please verify all information with the HR department.*
"""
    return markdown_output

@app.post("/api/hr-extract")
async def extract_hr_document(file: UploadFile = File(...)):
    logger.info("Processing HR letter for detailed analysis")
    try:
        if not file or not file.filename:
            logger.warning("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
            
        contents = await file.read()
        logger.debug(f"File size: {len(contents)} bytes")
        
        # If PDF, try text extraction first, then fall back to OCR if needed
        if file.filename and file.filename.lower().endswith('.pdf'):
            # Try text extraction first
            pdf_file = io.BytesIO(contents)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            # If no text found, use OCR
            if not text.strip():
                logger.info("No text found in PDF, falling back to OCR")
                images = convert_from_bytes(contents)
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
        else:
            # For image files
            logger.info("Processing image file with OCR")
            image = Image.open(io.BytesIO(contents))
            text = pytesseract.image_to_string(image)
        
        # Process with LangChain for detailed HR letter analysis
        llm = ChatOpenAI(
            temperature=0,
            api_key=api_key,
            model="gpt-4"
        )
        
        # Updated prompt to ensure JSON response
        analysis_prompt = f"""
        You are a precise HR document analyzer. Analyze the following HR letter and extract the required information.
        Return ONLY a valid JSON object with the following structure, no additional text or explanation:

        HR Letter Content:
        {text}

        Required JSON structure:
        {{
            "employee_name": "extracted name or null",
            "employee_id": "extracted ID or null",
            "department": "extracted department or null",
            "position": "extracted position or null",
            "joining_date": "extracted joining date or null",
            "gross_salary": "extracted gross salary (number only) or null",
            "net_salary": "extracted net salary (number only) or null",
            "letter_date": "date in YYYY-MM-DD format or null",
            "has_stamp": boolean,
            "is_valid": boolean,
            "validity_reason": "explanation string",
            "confidence_score": number between 0 and 1
        }}

        Rules:
        1. Format dates as YYYY-MM-DD
        2. Remove currency symbols from salary
        3. Return only the JSON object, no other text
        4. Ensure all boolean values are true/false (lowercase)
        5. Ensure confidence_score is a number between 0 and 1
        """

        # Get structured analysis from LLM
        logger.info("Generating structured analysis...")
        analysis_response = llm.predict(analysis_prompt)
        
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {analysis_response}")
            
            # Clean the response string
            cleaned_response = analysis_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse the JSON
            analysis_data = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = [
                "employee_name", "employee_id", "department", "position",
                "joining_date", "gross_salary", "net_salary", "letter_date",
                "has_stamp", "is_valid", "validity_reason", "confidence_score"
            ]
            
            for field in required_fields:
                if field not in analysis_data:
                    analysis_data[field] = None
            
            # Additional validation
            today = datetime.now()
            if analysis_data.get("letter_date"):
                try:
                    letter_date = datetime.strptime(analysis_data["letter_date"], "%Y-%m-%d")
                    three_months_ago = today - timedelta(days=90)
                    analysis_data["is_valid"] = letter_date >= three_months_ago
                    if not analysis_data["is_valid"]:
                        analysis_data["validity_reason"] = "Letter is older than 3 months"
                except ValueError:
                    logger.warning(f"Invalid date format: {analysis_data['letter_date']}")
                    analysis_data["letter_date"] = None
            
            # Format the response using markdown
            formatted_output = format_hr_letter_analysis(text, analysis_data)
            
            return {
                "markdown_output": formatted_output,
                "raw_analysis": analysis_data,
                "confidence": analysis_data.get("confidence_score", 0.95)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Problematic response: {analysis_response}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing the document analysis: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error in HR letter analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/doc-upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        contents = await file.read()
        file_id = str(uuid.uuid4())
        
        # Get file extension safely
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ''
        
        # Save file temporarily
        temp_path = f"temp_{file_id}{file_extension}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Load and split document
        if file.filename and file.filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)
            
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        vector_store.add_documents(splits, ids=[file_id])
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "id": file_id,
            "filename": file.filename,
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize chat chain
        llm = ChatOpenAI(
            temperature=0,
            api_key=api_key
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"question": request.message, "chat_history": []})
        
        # Check if response contains diagram request
        if "diagram" in request.message.lower() or "graph" in request.message.lower():
            # Generate diagram using graphviz
            dot = graphviz.Digraph(comment='Generated Diagram')
            # Add diagram generation logic based on the context
            # This is a simplified example
            dot.node('A', 'Concept A')
            dot.node('B', 'Concept B')
            dot.edge('A', 'B')
            
            # Create diagrams directory if it doesn't exist
            os.makedirs("diagrams", exist_ok=True)
            
            # Save diagram
            diagram_path = f"diagrams/diagram_{uuid.uuid4()}.png"
            dot.render(diagram_path, format='png')
            
            # Include diagram path in response
            return {
                "response": result["answer"],
                "diagram": diagram_path
            }
        
        return {
            "response": result["answer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

async def watch_files():
    logger.info("Starting file watcher...")
    async for changes in awatch("backend"):
        logger.info(f"Detected changes in files: {changes}")
        # You could add custom reload logic here if needed

def start_server():
    logger.info("Starting FastAPI server...")
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    server = uvicorn.Server(config)
    return server

if __name__ == "__main__":
    # Setup logging for uvicorn
    logging.getLogger("uvicorn").handlers = logger.handlers
    
    # Create and run the server
    server = start_server()
    
    # Run the server and file watcher
    asyncio.run(server.serve())