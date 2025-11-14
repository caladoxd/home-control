import chromadb
from chromadb.utils import embedding_functions
from google import genai
import os, json

class RoutineMemory:
    def __init__(self, collection_name="routines"):
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embed_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "text-embedding-004"

    def embed(self, text: str):
        emb = self.embed_client.models.embed_content(
            model=self.model,
            contents=text
        ).embedding
        return emb

    def store_routine(self, name: str, description: str, tool_sequence: list[dict]):
        embedding = self.embed(description)
        self.collection.add(
            ids=[name],
            embeddings=[embedding],
            documents=[description],
            metadatas=[{"tool_sequence": json.dumps(tool_sequence)}]
        )

    def find_similar_routine(self, query: str, threshold: float = 0.8):
        query_emb = self.embed(query)
        results = self.collection.query(query_embeddings=[query_emb], n_results=1)
        if results["distances"] and results["distances"][0][0] < (1 - threshold):
            metadata = results["metadatas"][0][0]
            return json.loads(metadata["tool_sequence"])
        return None
