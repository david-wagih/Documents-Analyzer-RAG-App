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
from graphviz import Digraph
from langchain.memory import ConversationBufferMemory

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
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    logger.info(f"Received document upload request: {file.filename}")
    try:
        if not file or not file.filename:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
            
        contents = await file.read()
        file_id = str(uuid.uuid4())
        
        # Get file extension safely
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        logger.info(f"Processing file with extension: {file_extension}")
        
        # Create temp directory if it doesn't exist
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file temporarily with safe naming
        temp_path = os.path.join(temp_dir, f"temp_{file_id}{file_extension}")
        logger.info(f"Saving file temporarily at: {temp_path}")
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        try:
            # Load and split document based on file type
            logger.info("Processing document...")
            if file_extension in ['.pdf']:
                logger.info("Loading PDF document")
                loader = PyPDFLoader(temp_path)
            elif file_extension in ['.txt', '.md']:
                logger.info("Loading text document")
                loader = TextLoader(temp_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}"
                )
                
            # Load the document
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages/sections")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text splits")
            
            # Create embeddings and store in ChromaDB
            try:
                logger.info("Creating embeddings and storing in ChromaDB...")
                # Generate unique IDs for each chunk
                chunk_ids = [f"{file_id}_{i}" for i in range(len(splits))]
                
                # Extract texts and metadata
                texts = [doc.page_content for doc in splits]
                metadatas = [
                    {
                        **doc.metadata,
                        "file_id": file_id,
                        "chunk_id": chunk_id,
                        "filename": file.filename,
                        "created_at": datetime.now().isoformat()
                    } 
                    for doc, chunk_id in zip(splits, chunk_ids)
                ]
                
                # Add to vector store
                vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
                
                logger.info(f"Successfully stored {len(splits)} chunks in ChromaDB")
                
                return {
                    "id": file_id,
                    "filename": file.filename,
                    "status": "ready",
                    "chunks": len(splits),
                    "metadata": {
                        "file_type": file_extension,
                        "total_chunks": len(splits),
                        "created_at": datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                logger.error(f"Error storing in ChromaDB: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to store document embeddings"
                )
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Cleaned up temporary file")
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.message}")
        logger.info(f"Document IDs: {request.documentIds}")

        llm = ChatOpenAI(
            temperature=0,
            api_key=api_key,
            model="gpt-4"
        )

        retriever = vector_store.as_retriever(
            
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        graph_keywords = ['draw', 'diagram', 'graph', 'visualize', 'flowchart', 'sequence']
        is_graph_request = any(keyword in request.message.lower() for keyword in graph_keywords)

        if is_graph_request:
            context_docs = retriever.get_relevant_documents(request.message)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            logger.info(f"Found {len(context_docs)} relevant documents for diagram")

            # Analyze request clarity
            clarity_prompt = f"""
            Analyze if the diagram request is clear enough to determine a diagram type.
            Request: "{request.message}"
            Context available about: {[doc.metadata.get('filename', 'Unnamed document') for doc in context_docs]}

            Return a JSON object with these fields:
            {{
                "is_clear": boolean,
                "suggested_types": list of applicable diagram types,
                "reason": explanation string,
                "needs_clarification": what needs to be clarified (if unclear)
            }}
            """
            
            clarity_response = llm.predict(clarity_prompt)
            clarity_data = json.loads(clarity_response)

            if not clarity_data.get("is_clear", False):
                # Return a response asking for clarification
                diagram_types_explanation = """
                Available diagram types:
                1. flowchart - For processes, workflows, and decision trees
                2. sequenceDiagram - For interactions between components/systems
                3. classDiagram - For structure and relationships between entities
                4. stateDiagram - For state machines and transitions
                5. erDiagram - For entity-relationship models
                """
                
                suggested_types = clarity_data.get("suggested_types", [])
                suggestion_text = ""
                if suggested_types:
                    suggestion_text = f"\n\nBased on your request, you might be interested in: {', '.join(suggested_types)}"

                clarification_response = {
                    "response": f"I'd be happy to create a diagram, but I need a bit more clarification. {clarity_data.get('reason', '')}\n\n{diagram_types_explanation}{suggestion_text}\n\nCould you please specify which type of diagram you'd like to see and what specific aspect you want to visualize?",
                    "type": "clarification",
                    "suggested_types": suggested_types,
                    "needs_clarification": clarity_data.get("needs_clarification"),
                    "sources": [
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        }
                        for doc in context_docs
                    ]
                }
                return clarification_response

            # Continue with diagram generation if request is clear
            diagram_type = clarity_data.get("suggested_types", ["flowchart"])[0]
            logger.info(f"Selected diagram type: {diagram_type}")

            # Create diagram generation prompt based on type
            diagram_prompt = f"""
            Based on the following context and request, generate a mermaid.js diagram.
            Use the determined diagram type: {diagram_type}

            Context from documents:
            {context_text}

            Request: {request.message}

            Rules:
            1. Use proper mermaid.js syntax for {diagram_type}
            2. Include only relevant information from the context
            3. Keep the diagram clean and readable
            4. Add clear labels and descriptions
            5. Use proper directional flow
            6. Include any relevant relationships or connections
            7. Add comments to explain complex parts

            Return ONLY the mermaid code wrapped in ```mermaid and ``` tags.
            """

            # Get diagram code from LLM
            diagram_response = llm.predict(diagram_prompt)
            
            # Extract mermaid code
            mermaid_match = re.search(r'```mermaid\n(.*?)```', diagram_response, re.DOTALL)
            if mermaid_match:
                mermaid_code = mermaid_match.group(1).strip()
                
                # Generate explanation for the diagram
                explanation_prompt = f"""
                Explain the diagram you just created. Include:
                1. What the diagram represents
                2. Key components and their relationships
                3. How it relates to the original request
                4. Important details to note
                
                Keep the explanation clear and concise.
                """
                
                explanation = llm.predict(explanation_prompt)
                
                return {
                    "response": explanation,
                    "mermaid": mermaid_code,
                    "type": "diagram",
                    "diagram_type": diagram_type,
                    "sources": [
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        }
                        for doc in context_docs
                    ]
                }
            else:
                raise ValueError("Failed to generate valid diagram code")

        else:
            # Regular chat response handling
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff"
            )
            
            result = qa_chain({
                "question": request.message,
                "chat_history": []
            })
            
            # Log retrieved documents for debugging
            logger.info(f"Retrieved {len(result.get('source_documents', []))} source documents")
            for i, doc in enumerate(result.get('source_documents', [])):
                logger.info(f"Document {i + 1} content preview: {doc.page_content[:100]}...")

            # Format response with source information
            response = {
                "response": result["answer"],
                "type": "text",
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result.get('source_documents', [])
                ],
                "has_context": bool(result.get('source_documents', []))
            }

            if not response["has_context"]:
                logger.warning("No relevant context found for the query")
                response["response"] = "I apologize, but I couldn't find relevant information in the provided documents to answer your question accurately."

            return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
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