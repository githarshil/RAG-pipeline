import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables (.env)
load_dotenv()

pdf_path = Path(__file__).parent / "PDF-Guide-Node-Andrew-Mead-v3.pdf"

# 1. Loading the PDF file
print(f"Loading document from {pdf_path.name}...")
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

# 2. Splitting the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(documents=docs)
print(f"Split into {len(chunks)} chunks.")

# 3. Vector embeddings using Gemini SDK
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# 4. Ingest chunks into Qdrant Vector Store
print("Generating embeddings and indexing into Qdrant...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print(f"Indexing complete! Successfully indexed {len(chunks)} chunks into Qdrant collection 'learning_rag'.")