from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Setup LLM and Embedding model
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load database
db = chromadb.PersistentClient(path="./storage/chroma")
chroma_collection = db.get_or_create_collection("lecture_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# Initialize chatbot
chat_engine = index.as_chat_engine(
    chat_mode="context", 
    system_prompt="You are a course tutor. " \
    "your job includes clarifying the user's doubts on course content, explaining concepts in an understanding manner without missing out on key details and info, quizing the user on their notes when asked to. " \
    "Ignore headers, footers and any other miscellaneous details that may appear in the page (copyrights, page number, etc.)" \
    "Use Socratic questioning: lead them to the answer rather than giving it away."
)

print("RAG system ready! You can start chatting (type 'exit' to stop).")

while True:
    text_input = input("\nUser: ")
    if text_input.lower() == "exit":
        break
    
    # stream_chat to see typing effect
    response = chat_engine.stream_chat(text_input)
    
    print("Tutor: ", end="")
    for token in response.response_gen:
        print(token, end="", flush=True)
    print("\n")