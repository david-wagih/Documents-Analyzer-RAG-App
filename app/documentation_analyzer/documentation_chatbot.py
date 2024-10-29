import os
import gradio as gr
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document
import ollama
from tempfile import NamedTemporaryFile


def load_and_process_document(url_or_path, progress=gr.Progress()):
    progress(0.0, desc="Starting document processing")
    
    # Load documents (same as before)
    if url_or_path.lower().startswith('http'):
        loader = UnstructuredURLLoader(urls=[url_or_path])
        documents = loader.load()
    else:
        documents = read_file(url_or_path)
    
    progress(0.3, desc="Document loaded")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if not documents:
        raise ValueError("No documents were processed. Please check your input.")
    
    progress(0.5, desc="Creating embeddings")
    
    # Updated ChromaDB initialization
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    progress(1.0, desc="Document processing complete")
    
    return vectorstore

def create_conversational_agent(vectorstore):
    def get_response(question: str, chat_history: list) -> str:
        # Format chat history for context
        chat_context = "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in chat_history])
        
        # Get relevant documents from vectorstore
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are a technical documentation assistant. Use the following context to answer the question.
        If you don't know the answer, just say that you don't know. Keep the answer focused and technical.
        
        Context:
        {context}
        
        Chat History:
        {chat_context}
        
        Question: {question}
        
        Answer:"""
        
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'system',
                'content': 'You are a technical documentation expert specializing in software engineering documentation.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        return response['message']['content']

    return get_response

def respond(message, chat_history):
    if not hasattr(respond, 'agent'):
        return "", [{"role": "assistant", "content": "Please load a document first."}]
    
    # Convert chat history to the format expected by the agent
    formatted_history = []
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            formatted_history.append((
                chat_history[i]["content"],
                chat_history[i + 1]["content"]
            ))
    
    # Get response from agent
    response = respond.agent(message, formatted_history)
    
    # Update chat history
    chat_history.append({"role": "human", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    
    return "", chat_history

def gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# Technical Documentation Chatbot")
        
        with gr.Row():
            url_input = gr.Textbox(label="Enter Documentation URL")
            file_upload = gr.File(label="Or Upload Document", file_types=[".pdf", ".txt"])
        
        scan_button = gr.Button("Process Document")
        status_box = gr.Textbox(label="Status", interactive=False)
        chatbot = gr.Chatbot(label="Conversation", type="messages")
        
        with gr.Row():
            msg = gr.Textbox(label="Your Question")  # Remove interactive=False
            send_button = gr.Button("Send", interactive=True)  # Set to True by default

        def start_scan(url_or_path, uploaded_file):
            try:
                if uploaded_file is not None:
                    if isinstance(uploaded_file, str):
                        file_path = uploaded_file
                    else:
                        file_path = uploaded_file.name
                elif url_or_path:
                    file_path = url_or_path
                else:
                    return "Please provide a URL or upload a document.", [], []

                vectorstore = load_and_process_document(file_path)
                respond.agent = create_conversational_agent(vectorstore)
                
                return (
                    "Document processed successfully. You can start asking questions.",
                    [],  # Clear chat history
                    []   # Clear any existing message
                )
            except Exception as e:
                return f"Error: {str(e)}", [], []

        scan_button.click(
            start_scan,
            inputs=[url_input, file_upload],
            outputs=[status_box, chatbot, msg]  # Update outputs
        )
        
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_button.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    iface.launch(share=False)

def read_file(file_path):
    """Read content from PDF or text files"""
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
        return loader.load()
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")

def upload_file(file):
    """Handle file upload and return the temporary file path"""
    if isinstance(file, str):
        return file, os.path.basename(file)
        
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        if hasattr(file, 'name'):
            # If it's a file path
            with open(file.name, 'rb') as f:
                tmp_file.write(f.read())
        else:
            # If it's a file-like object
            file.seek(0)
            tmp_file.write(file.read())
        return tmp_file.name, os.path.basename(file.name)

if __name__ == "__main__":
    gradio_interface()
