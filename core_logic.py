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
        Settings.llm = Ollama(model="gemma3:1b", request_timeout=600.0, temperature=0.1)
        
        # Vector DB Connection
        self.client = qdrant_client.QdrantClient(path="./qdrant_db")
        self.vector_store = QdrantVectorStore(client=self.client, collection_name="it_knowledge_base")
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)


    def triage_classify(self, query):
        # Classify query into a category for speedy retrieval
        router_prompt = PromptTemplate("""
        SYSTEM: You are a strict router. Categorize the user's issue.
        - If it mentions WiFi, Internet, IP, DNS, DHCP, Connection -> networking 
        - If it mentions Printer, Monitor, Screen, Laptop, Hardware -> hardware
        - If it mentions Password, Phishing, Login, Access, MFA -> security
        - Otherwise -> general
                                    
        Question: {query_text}
        Output only the single word:""")
        
        response = Settings.llm.predict(router_prompt, query_text=query).strip().lower()
        for category in ['networking', 'hardware', 'security']:
            if category in response:
                return category
        return 'general'

    def run_query(self, query_text, category):
        # Apply Metadata Filter
        filters = MetadataFilters(filters=[ExactMatchFilter(key="category", value=category)])
        
        # System Prompt to limit knowledge scope to knowledge base only
        S_PROMPT = (
            f"You are a technician for the {category} department. "
            "Use ONLY the provided context. If the answer is not there, "
            "say: 'I do not have documentation for this in the knowledge base.' "
            "DO NOT give general advice.")

        # Initialize engine
        engine = self.index.as_query_engine(
            filters=filters, 
            similarity_top_k=3, 
            streaming=True,
            system_prompt=S_PROMPT
        )
        
        # Response retrieval
        return engine.query(query_text)
        
        # Verify sources of response 
        """print("\n\n SOURCES:")
        if not response.source_nodes:
            print("- No documents retrieved.")
        for node in response.source_nodes:
            fname = node.node.metadata.get('file_name', 'Unknown Source')
            print(f"- {fname} (Match Score: {node.score:.2f})")
        print("-----------------------------------------------------------")"""


# --- TESTER ---
if __name__ == "__main__":
    bot = ITAssistant()

    # Test Question
    #bot.run_query("i got a phone call asking for my SIN number, is this legit?")