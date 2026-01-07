import qdrant_client
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

class ITAssistant:
    def __init__(self):
        # Initialize models
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        Settings.llm = Ollama(model="llama3.2", request_timeout=600.0, temperature=0.0)
        
        # Vector DB Connection
        self.client = qdrant_client.QdrantClient(path="./qdrant_db")
        self.vector_store = QdrantVectorStore(client=self.client, collection_name="it_knowledge_base")
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)


    def triage_intent(self, query):
        # Classify query into a category for speedy retrieval
        router_prompt = PromptTemplate("""
        SYSTEM: You are a strict IT Triage specialist. 
        Categorize the request into: 'networking', 'hardware', 'security', or 'general'.
        RULES:
        - WiFi/VPN/Internet/IP/Cables -> networking
        - Printer/Monitor/Screen/Laptop/Keyboard -> hardware
        - Password/Phishing/Login/Access/Link -> security
        - Everything else -> general
                                       
        Question: {query_text}
        Category:""")
        
        response = Settings.llm.predict(router_prompt, query_text=query).strip().lower()
        
        # Make sure catgory is valid
        for category in ['networking', 'hardware', 'security']:
            if category in response:
                return category
        return 'general'

    def run_query(self, query_text):
        # Get Category
        category = self.triage_intent(query_text)
        print(f"[DOMAIN]: {category.upper()}")

        # Apply Metadata Filter
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="category", value=category)
        ])
        
        # System Prompt to limit knowledge scope to knowledge base only
        S_PROMPT = (
            f"You are a professional IT Support specialist. "
            f"Your knowledge is strictly limited to the provided {category} documentation. "
            "If the answer is not in the context, say 'I cannot find that in the official knowledge base.' "
            "Do not use external knowledge or make up security advice."
        )

        # Initialize engine
        engine = self.index.as_query_engine(
            filters=filters, 
            similarity_top_k=3, 
            streaming=True,
            system_prompt=S_PROMPT
        )
        
        # Response retrieval
        response = engine.query(query_text)
        print("IT Assistant:")
        response.print_response_stream()
        
        # Verify sources of response 
        print("\n\n SOURCES:")
        if not response.source_nodes:
            print("- No documents retrieved.")
        for node in response.source_nodes:
            fname = node.node.metadata.get('file_name', 'Unknown Source')
            print(f"- {fname} (Match Score: {node.score:.2f})")
        print("-----------------------------------------------------------")


# --- TESTER ---
if __name__ == "__main__":
    bot = ITAssistant()

    # Test Question
    #bot.run_query("i got a phone call asking for my SIN number, is this legit?")