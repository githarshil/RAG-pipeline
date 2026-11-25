from h11._abnf import chunk_size
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdf_path = Path(__file__).parent / "PDF-Guide-Node-Andrew-Mead-v3.pdf"
# loading the file into the prog
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
# splitting the doc into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
text_splitter.split_documents(documents=docs)