import os
import gc
import time
import uuid
import uvicorn
import logging
import shutil
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# Setup logging to a file
logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

mcp = FastMCP("rag")
llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY
)

persist_directory = None  # Will be set per PDF
current_chroma_dir = None  # Track current chroma dir for cleanup
retriever = None
qa_chain = None
last_loaded_pdf = None
vectordb = None


# Clean chroma store
def clear_previous_embeddings():
    global vectordb, current_chroma_dir

    if vectordb is not None:
        try:
            vectordb._collection = None
            vectordb = None
        except Exception as e:
            print("Error clearing vectordb:", e)

    gc.collect()
    time.sleep(0.5)

    if current_chroma_dir and os.path.exists(current_chroma_dir):
        try:
            shutil.rmtree(current_chroma_dir)
            print(f"Chroma store {current_chroma_dir} cleared.")
        except Exception as e:
            print(f"Failed to delete {current_chroma_dir}: {e}")
    current_chroma_dir = None


# Load and split PDF into chunks
def load_pdf_chunks(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(documents)


# Embed and store in Chroma
def store_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    return vectordb


def setup_pdf_knowledge(pdf_path):
    global retriever, qa_chain, persist_directory, current_chroma_dir
    try:
        clear_previous_embeddings()
        # Generate a new unique chroma directory for this PDF
        new_chroma_dir = os.path.join("chroma_store", str(uuid.uuid4()))
        persist_directory = new_chroma_dir
        current_chroma_dir = new_chroma_dir
        docs = load_pdf_chunks(pdf_path)
        vectordb = store_embeddings(docs)
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        logging.info("PDF loaded and embeddings stored successfully...")
    except Exception as e:
        logging.info(f"Failed to process PDF: {str(e)}")


@mcp.tool()
def load_pdf_tool(pdf_name: str) -> str:
    """
    Loads the given PDF into memory and prepares it for question answering.
    Clears previous PDF state and embeddings.
    """
    global last_loaded_pdf, retriever, qa_chain
    last_loaded_pdf = os.path.join("uploads", pdf_name)
    if not os.path.exists(last_loaded_pdf):
        return f"Error: PDF '{pdf_name}' not found in server 'uploads/' directory."
    
    # Reset all previous state
    retriever = None
    qa_chain = None
    clear_previous_embeddings()
    setup_pdf_knowledge(last_loaded_pdf)

    return f"PDF '{pdf_name}' loaded and previous state cleared successfully."


@mcp.tool()
def rag_tool(query: str) -> str:
    """
    Fetch information from PDF knowledge base using query.
    """
    global last_loaded_pdf, qa_chain

    logging.info("Entering into Rag tool...")
    print("Q: ", query)

    if last_loaded_pdf is None:
        return "Error: No PDF has been loaded. Please upload a PDF first."

    if not os.path.exists(last_loaded_pdf):
        return f"Error: PDF file not found at {last_loaded_pdf}"

    if qa_chain is None:
        print("QA Chain not initialized")

    print("Using query:", query)
    result = qa_chain.invoke(query)
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or str(result)
    else:
        answer = str(result)
    print("Final: ", answer)
    return answer


app = mcp.streamable_http_app

if __name__ == "__main__":
    uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=8000)
