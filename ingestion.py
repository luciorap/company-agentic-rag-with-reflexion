import os
import sys
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Web Documents
print("Loading web docs...", flush=True)
urls = ["https://www.crunchbase.com", "https://www.statista.com", "https://www.apollo.io"]
web_loader = WebBaseLoader(urls, header_template={"User-Agent": "Mozilla/5.0"})
web_docs = web_loader.load()
print(f"Loaded {len(web_docs)} web documents", flush=True)

# Load Internal Documents
internal_docs = []
if os.path.exists("./internal_docs"):
    print("Loading internal docs...", flush=True)
    pdf_loader = DirectoryLoader("./internal_docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docx_loader = DirectoryLoader("./internal_docs", glob="**/*.docx", loader_cls=Docx2txtLoader)
    internal_docs = pdf_loader.load() + docx_loader.load()
    print(f"Loaded {len(internal_docs)} internal documents", flush=True)
else:
    print("No internal_docs folder found", flush=True)

# Process & Index 
all_docs = web_docs + internal_docs
print(f"Total documents: {len(all_docs)}", flush=True)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=40)
doc_splits = text_splitter.split_documents(all_docs)
print(f"Split into {len(doc_splits)} chunks", flush=True)

persist_directory = "./.chroma"

# Vectorstore Logic
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    print("Creating new vectorstore...", flush=True)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_function,
        persist_directory=persist_directory,
    )
    print(f"Created vectorstore with {vectorstore._collection.count()} documents", flush=True)
else:
    print("Loading existing vectorstore...", flush=True)
    vectorstore = Chroma(
        collection_name="rag-chroma",
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )

retriever = vectorstore.as_retriever()
print("Done!", flush=True)
