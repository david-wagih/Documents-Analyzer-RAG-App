import os
import ssl
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import warnings
from urllib3.exceptions import InsecureRequestWarning
import tempfile
import requests
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import uuid
import shutil
from tqdm import tqdm


# Suppress InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# Disable SSL verification (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context


load_dotenv()

def process_uploaded_file(file):
    if file is None:
        return None
    if hasattr(file, 'data'):
        # file is a Gradio 'NamedString' object
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.data)
            return temp_file.name
    else:
        raise ValueError("Unsupported file object")


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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    doc_chunks = text_splitter.create_documents(raw_text)
    return doc_chunks

def read_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='ara+eng')
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "،", " "],  # Added Arabic comma
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    doc_chunks = text_splitter.create_documents([text])
    return doc_chunks

# Custom UnstructuredURLLoader that disables SSL verification
class UnstructuredURLLoaderInsecure(UnstructuredURLLoader):
    def _fetch_content(self, url):
        response = requests.get(url, verify=False)
        response.raise_for_status()
        return response.content

def load_and_process_document(url_or_path, progress=gr.Progress()):
    progress(0, desc="Starting document processing")
    
    # Determine if the input is a URL or a file path
    if url_or_path.lower().startswith('http'):
        # Load content from URL
        loader = UnstructuredURLLoader(urls=[url_or_path])
        documents = loader.load()
    else:
        # Load content from file
        documents = read_file(url_or_path)
    
    progress(0.3, desc="Document loaded")
    
    embeddings = OpenAIEmbeddings()
    
    # Check if docs is empty
    if not documents:
        raise ValueError("No documents were processed. Please check your input.")
    
    progress(0.5, desc="Creating embeddings")
    
    # Create embeddings with progress updates
    total_docs = len(documents)
    vectorstore = FAISS.from_documents(
        tqdm(documents, desc="Processing documents", total=total_docs),
        embeddings
    )
    
    progress(1.0, desc="Document processing complete")
    
    return vectorstore

def create_conversational_agent(vectorstore):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found.")
    
    llm = ChatOpenAI(temperature=0, api_key=api_key)
    retriever = vectorstore.as_retriever()

    # Create the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Define the graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the function that calls the model
    def call_model(state: MessagesState):
        question = state["messages"][-1].content
        chat_history = state.get("chat_history", [])
        result = qa_chain.invoke({"question": question, "chat_history": chat_history})
        response = AIMessage(content=result["answer"])
        return {"messages": [response], "chat_history": chat_history + [(question, result["answer"])]}

    # Define the graph structure
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app

def chatbot_interface(url_or_path, progress=gr.Progress()):
    vectorstore = load_and_process_document(url_or_path, progress)
    qa_app = create_conversational_agent(vectorstore)
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}
    return qa_app, config


def gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# Document Chatbot")
        
        with gr.Row():
            url_input = gr.Textbox(label="Enter Documentation URL or PDF Path")
            file_upload = gr.File(label="Or Upload PDF", file_types=[".pdf"])
        
        scan_button = gr.Button("Scan Document")
        
        status_box = gr.Textbox(label="Status", interactive=False)
        chatbot = gr.Chatbot(label="Conversation", type="messages")
        
        with gr.Row():
            msg = gr.Textbox(label="Your Question", interactive=False)
            send_button = gr.Button("Send", interactive=False)
        
        state = gr.State()
        thread_config = gr.State()

        def start_scan(url_or_path, uploaded_file, progress=gr.Progress()):
            try:
                if uploaded_file is not None:
                    file_path, _ = upload_file(uploaded_file)
                elif url_or_path:
                    file_path = url_or_path
                else:
                    return (
                        "Please provide a URL or upload a PDF file.",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        []  # Clear chat history
                    )

                progress(0.1, desc="Starting document processing")
                qa_app, config = chatbot_interface(file_path, progress)
                progress(1.0, desc="Document processing complete")
                
                state.value = qa_app
                thread_config.value = config
                msg.interactive = True
                send_button.interactive = True
                return (
                    "Document loaded and embedded. You can start asking questions now.",
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    []  # Clear chat history
                )
            except Exception as e:
                return (
                    f"Error: {str(e)}",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    []  # Clear chat history
                )

        def respond(message, chat_history):
            qa_app = state.value
            config = thread_config.value
            if qa_app is None:
                return "", [{"role": "assistant", "content": "Please load a document first."}]
            
            # Convert chat history to the format expected by ConversationalRetrievalChain
            lc_chat_history = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    human_msg = chat_history[i]["content"]
                    ai_msg = chat_history[i + 1]["content"]
                    lc_chat_history.append((human_msg, ai_msg))
            
            # Prepare the input for the qa_app
            input_data = {
                "messages": [HumanMessage(content=message)],
                "chat_history": lc_chat_history
            }
            
            for event in qa_app.stream(input_data, config, stream_mode="values"):
                response = event["messages"][-1].content
            
            # Update chat history with new messages
            chat_history.append({"role": "human", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            
            return "", chat_history

        scan_button.click(start_scan, inputs=[url_input, file_upload], outputs=[status_box, msg, send_button, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_button.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    iface.launch(share=True)

if __name__ == "__main__":
    gradio_interface()
