from langchain_community.document_loaders import url
from langchain_qdrant import qdrant
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag",
)
# user input
user_query = input("ask something: ")

# relevent chunks from vector db
search_result = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join(
    [
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label') or result.metadata.get('page')}\nFile Location: {result.metadata.get('source')}"
        for result in search_result
    ]
)
SYSTEM_PROMPT = f"""
You are a helpful ai assistant who answers based on the available context retrieved from a pdf file along with page_contents and page number.
Your should only answer based on the context and navigate the user to open the right page number
context: {context}"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)

response = llm.invoke(
    [
        ("system", SYSTEM_PROMPT),
        ("human", user_query),
    ]
)

print(response.content)

