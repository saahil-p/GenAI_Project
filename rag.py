import os
import torch
import hashlib
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import per deprecation warning

# Try to import the new Chroma package; if unavailable, fall back to the legacy import.
try:
    from langchain_chroma import Chroma  # New recommended import
    print("Using new langchain_chroma package.")
except ImportError:
    from langchain.vectorstores import Chroma  # Fallback to legacy import
    print("langchain_chroma not installed. Falling back to legacy langchain.vectorstores.Chroma.")

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate

# Enable Apple Metal (MPS) acceleration if available on your M1 device.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_device("mps")
    print("Using MPS for acceleration.")
else:
    device = torch.device("cpu")
    print("MPS not available. Falling back to CPU.")

# Define paths
TXT_DIRECTORY = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG"
CHROMA_PATH = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG/chroma_db"
COMBINED_TXT_PATH = os.path.join(TXT_DIRECTORY, "combined.txt")
HASH_FILE_PATH = os.path.join(TXT_DIRECTORY, "combined_txt.hash")

# Use Flan-T5-Small for fast local inference
MODEL_ID = "google/flan-t5-small"

# Global variable to hold the Chroma database instance for reuse.
db = None

def compute_file_hash(file_path):
    """Compute the MD5 hash of a file to detect changes."""
    if not os.path.exists(file_path):
        return None
    
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_file_hash(file_path, hash_file_path):
    """Save the current hash of the file."""
    file_hash = compute_file_hash(file_path)
    if file_hash:
        with open(hash_file_path, 'w') as f:
            f.write(file_hash)
    return file_hash

def load_saved_hash(hash_file_path):
    """Load the previously saved hash if it exists."""
    if os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as f:
            return f.read().strip()
    return None

def has_file_changed(file_path, hash_file_path):
    """Check if the file has changed by comparing hashes."""
    saved_hash = load_saved_hash(hash_file_path)
    current_hash = compute_file_hash(file_path)
    
    if not saved_hash or not current_hash:
        return True
    
    return saved_hash != current_hash

def load_docs() -> list[Document]:
    """Load documents from the specified directory."""
    loader = DirectoryLoader(TXT_DIRECTORY, glob="*.txt", loader_cls=TextLoader)
    return loader.load()

def split_docs(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for faster embedding and retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,        # Reduced chunk size for faster processing
        chunk_overlap=50,      # Reduced overlap
        length_function=lambda x: len(x.split())  # Using word count for splitting
    )
    return text_splitter.split_documents(documents)

def get_embedding_fn():
    """Create an embedding function using a lightweight, efficient model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def add_to_chroma(chunks: list[Document]):
    """Add document chunks to the Chroma vector database and persist."""
    global db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_fn())
    
    # Assign an ID to each chunk for tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"chunk_{i}"
    
    db.add_documents(chunks)
    db.persist()
    return db

def reset_chroma_db():
    """Delete the existing Chroma database to rebuild it from scratch."""
    global db
    db = None
    import shutil
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting existing vector database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

def get_chroma_db():
    """Retrieve or initialize the global ChromaDB instance, updating if source file has changed."""
    global db
    
    # Check if combined.txt has changed
    if os.path.exists(COMBINED_TXT_PATH) and has_file_changed(COMBINED_TXT_PATH, HASH_FILE_PATH):
        print(f"Changes detected in {COMBINED_TXT_PATH}. Rebuilding vector database...")
        reset_chroma_db()
        docs = load_docs()
        chunks = split_docs(docs)
        db = add_to_chroma(chunks)
        # Save the new hash after updating
        save_file_hash(COMBINED_TXT_PATH, HASH_FILE_PATH)
        return db
    
    # If no changes or the file doesn't exist, use existing DB or create a new one
    if db is None:
        if not os.path.exists(CHROMA_PATH):
            print("Creating new vector database...")
            docs = load_docs()
            chunks = split_docs(docs)
            db = add_to_chroma(chunks)
            # Save hash after initial creation if combined.txt exists
            if os.path.exists(COMBINED_TXT_PATH):
                save_file_hash(COMBINED_TXT_PATH, HASH_FILE_PATH)
        else:
            print("Using existing vector database...")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_fn())
    
    return db

def get_retriever(db):
    """Create a retriever from the vector database."""
    return db.as_retriever(search_kwargs={"k": 3})

def get_llm():
    """
    Initialize the Flan-T5-Small model for RAG.
    Flan-T5-Small is an encoder-decoder model that is much lighter (~80M parameters)
    and suitable for fast local inference on devices like the MacBook Air M1.
    """
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Set up the text2text-generation pipeline using greedy decoding for speed.
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=250,# Limit output length for faster response,
        min_new_tokens=10,
        do_sample=False  # Greedy decoding for deterministic and fast generation
    )
    
    return HuggingFacePipeline(pipeline=llm_pipeline)

def setup_qa_pipeline(db):
    """Set up the RAG pipeline with the Flan-T5-Small model and a custom prompt template."""
    retriever = get_retriever(db)
    llm = get_llm()
    
    prompt_template = """<|system|>
You are an expert assistant specialized in oil well extraction. Your task is to provide accurate, concise information based ONLY on the context provided.
If the context doesn't contain enough information to answer the question, you must say "I don't have enough information to answer this question."
Do not make up information or rely on prior knowledge not present in the context.
Respond in a clear, formal, and professional manner. 
<|user|>
Context information:
{context}

Based on this context, answer the following question: {question}
<|assistant|>
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

def run_rag_query(query_text: str) -> str:
    """
    Run a complete RAG query using the optimized Flan-T5-Small model.
    
    Args:
        query_text: The question to answer.
        
    Returns:
        The generated answer as a string.
    """
    db_instance = get_chroma_db()
    print("Setting up RAG pipeline with Flan-T5-Small...")
    qa_pipeline = setup_qa_pipeline(db_instance)
    
    print(f"Generating answer to: {query_text}")
    response = qa_pipeline.invoke({"query": query_text})
    
    return response['result']

if __name__ == "__main__":
    # Check if combined.txt has changed and update the database if needed
    get_chroma_db()
    
    # Ask a specific question about oil well extraction
    query = "What does abrupt increase in BSW mean?"
    
    answer = run_rag_query(query)
    answer = "Well is experiencing an abrupt increase in BSW." + answer
    
    print("\nGenerated Response:")
    print(answer)

    query = "What are the causes an abrupt increase in BSW in oil wells?"

    answer = run_rag_query(query)
    print("\nGenerated Response:\n")
    print(answer)

    # query = "What are the risk mitigation and fallback options for an abrupt increase in BSW in oil wells?"

    # answer = run_rag_query(query)
    # print("\nGenerated Response:\n")
    # print(answer)