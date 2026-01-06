import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# Model initialization
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def build_index():
    # 1. Load documents 
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 2. Setup ChromaDB (Local Storage)
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection("lecture_notes")
    
    # 3. Create the Index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Turn texts into vectors and store in ChromaDB
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("Success: Lectures indexed and saved to disk.")

if __name__ == "__main__":
    build_index()