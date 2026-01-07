import os
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# Embedding Model Setup
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def run_ingestion():
    if not os.path.exists("./data"):
        print("Error")
        return

    print("Loading documents...")
    # Use the folder name as the category metadata
    reader = SimpleDirectoryReader(
        input_dir="./data", 
        recursive=True, 
        file_metadata=lambda x: {
        "file_name": os.path.basename(x),
        "category": os.path.basename(os.path.dirname(x))}
    )
    documents = reader.load_data()

    print("Initializing Qdrant Vector DB...")
    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(client=client, collection_name="it_knowledge_base")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Creating Embeddings...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    print("Ingestion Complete. Database saved to ./qdrant_db")

if __name__ == "__main__":
    run_ingestion()