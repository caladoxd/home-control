#!/usr/bin/env python3
"""
ChromaDB Vector Database Service

This module provides a vector database service using ChromaDB for storing and retrieving
conversation history, tool descriptions, and other semantic data.

How ChromaDB Works:
1. **Embeddings**: Text is converted to numerical vectors (embeddings) using Google's text-embedding-004 model
2. **Storage**: These vectors are stored in ChromaDB collections
3. **Similarity Search**: When querying, your search text is also converted to an embedding,
   and ChromaDB finds the most similar vectors using cosine similarity
4. **Retrieval**: Returns the most relevant documents based on semantic similarity

Use Cases:
- Store conversation history for context retrieval
- Store tool descriptions for semantic search
- Store device information and commands
- Enable RAG (Retrieval Augmented Generation) for better AI responses
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import os
import json
import logging
from datetime import datetime
from google import genai

logger = logging.getLogger(__name__)


class GoogleEmbeddingFunction:
    """Custom embedding function using Google's genai.embed_content API with fallback to sentence transformers"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-004"):
        """
        Initialize Google embedding function with fallback
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Embedding model to use (default: text-embedding-004)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.use_google = bool(self.api_key)
        self.google_client = None
        self.fallback_function = None
        
        if self.use_google:
            try:
                self.google_client = genai.Client(api_key=self.api_key)
                # Test if API works by trying a simple call
                try:
                    test_result = self.google_client.models.embed_content(
                        model=self.model,
                        contents="test"
                    )
                    _ = test_result.embedding  # type: ignore
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a quota/429 error
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        logger.warning(f"Google embedding API quota exceeded or not available. Falling back to sentence transformers.")
                        self.use_google = False
                    else:
                        logger.warning(f"Google embedding API test failed: {e}. Falling back to sentence transformers.")
                        self.use_google = False
            except Exception as e:
                logger.warning(f"Failed to initialize Google embedding client: {e}. Falling back to sentence transformers.")
                self.use_google = False
        
        # Fallback to sentence transformers if Google API is not available
        if not self.use_google:
            logger.info("Using sentence transformers for embeddings (fallback)")
            self.fallback_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # Use fallback if Google API is not available
        if not self.use_google and self.fallback_function:
            fallback_result = self.fallback_function(input)
            # Convert to list of lists of floats
            return [[float(x) for x in emb] for emb in fallback_result] if fallback_result else [[0.0] * 384] * len(input)
        
        # Try Google API
        if not self.google_client:
            # Fallback if client is not initialized
            if self.fallback_function:
                fallback_result = self.fallback_function(input)
                return [[float(x) for x in emb] for emb in fallback_result] if fallback_result else [[0.0] * 384] * len(input)
            return [[0.0] * 768] * len(input)
        
        embeddings = []
        for text in input:
            try:
                result = self.google_client.models.embed_content(
                    model=self.model,
                    contents=text
                )
                # Access embedding - same pattern as memory.py
                embedding = result.embedding  # type: ignore
                if not embedding:
                    logger.warning(f"Empty embedding returned for text: {text[:50]}...")
                    # Fallback to sentence transformer for this text
                    if self.fallback_function:
                        fallback_emb = self.fallback_function([text])[0]
                        embedding = [float(x) for x in fallback_emb]
                    else:
                        embedding = [0.0] * 768
                
                embeddings.append(embedding)
            except Exception as e:
                error_str = str(e)
                # Check if it's a quota/429 error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    logger.warning(f"Google embedding API quota exceeded. Switching to fallback for remaining texts.")
                    self.use_google = False
                    if self.fallback_function:
                        # Process remaining texts with fallback
                        remaining_texts = input[len(embeddings):]
                        if remaining_texts:
                            fallback_embeddings = self.fallback_function(remaining_texts)
                            # Convert to list of floats
                            embeddings.extend([[float(x) for x in emb] for emb in fallback_embeddings])
                        # Also reprocess current text
                        fallback_emb = self.fallback_function([text])[0]
                        if embeddings:
                            embeddings[-1] = [float(x) for x in fallback_emb]
                        else:
                            embeddings = [[float(x) for x in fallback_emb]]
                        break
                
                logger.error(f"Error generating embedding: {e}")
                # Use fallback if available
                if self.fallback_function:
                    fallback_emb = self.fallback_function([text])[0]
                    embedding = [float(x) for x in fallback_emb]
                else:
                    embedding = [0.0] * 768
                embeddings.append(embedding)
        
        return embeddings


class VectorDBService:
    """Service for managing vector database operations with ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "conversations"):
        """
        Initialize ChromaDB service
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use Google's embedding API (text-embedding-004) with fallback to sentence transformers
        # Google API creates 768-dimensional vectors, fallback creates 384-dimensional vectors
        self.embedding_function = GoogleEmbeddingFunction(
            model="text-embedding-004"
        )
        
        # Get or create collection
        # Note: ChromaDB accepts callables for embedding_function, but type checker is strict
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,  # type: ignore
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        embedding_type = "Google embeddings" if self.embedding_function.use_google else "sentence transformers (fallback)"
        logger.info(f"Initialized ChromaDB collection '{collection_name}' at {persist_directory} using {embedding_type}")
    
    def add_conversation(
        self,
        user_input: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a conversation to the vector database
        
        Args:
            user_input: The user's input/command
            assistant_response: The assistant's response
            tool_calls: List of tool calls made during the conversation
            metadata: Additional metadata to store
            
        Returns:
            The ID of the added document
        """
        # Combine user input and response for better semantic search
        document_text = f"User: {user_input}\nAssistant: {assistant_response}"
        
        if tool_calls:
            tool_text = "\n".join([f"Tool: {tc.get('tool', 'unknown')}" for tc in tool_calls])
            document_text += f"\n{tool_text}"
        
        # Create metadata
        doc_metadata = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": assistant_response,
            **(metadata or {})
        }
        
        if tool_calls:
            doc_metadata["tool_calls"] = json.dumps(tool_calls)
        
        # Generate unique ID
        doc_id = f"conv_{datetime.now().timestamp()}_{hash(document_text) % 10000}"
        
        # Add to collection
        self.collection.add(
            documents=[document_text],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )
        
        logger.info(f"Added conversation to vector DB: {doc_id}")
        return doc_id
    
    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar conversations
        
        Args:
            query: The search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar conversations with metadata
        """
        where = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results and results.get('ids'):
            ids_list = results.get('ids', [[]])
            if ids_list and len(ids_list[0]) > 0:
                ids = ids_list[0]
                documents_list = results.get('documents', [[]])
                documents = documents_list[0] if documents_list and documents_list else []
                metadatas_list = results.get('metadatas', [[]])
                metadatas = metadatas_list[0] if metadatas_list and metadatas_list else []
                distances_list = results.get('distances')
                distances = distances_list[0] if distances_list and distances_list else []
                for i in range(len(ids)):
                    formatted_results.append({
                        "id": ids[i],
                        "document": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if distances and i < len(distances) else None
                    })
        
        logger.info(f"Found {len(formatted_results)} similar conversations for query: {query[:50]}...")
        return formatted_results
    
    def get_conversation_context(
        self,
        current_query: str,
        n_results: int = 10
    ) -> str:
        """
        Get relevant conversation context for a query
        
        Args:
            current_query: The current user query
            n_results: Number of similar conversations to retrieve
            
        Returns:
            Formatted context string for use in prompts
        """
        # Search with the original query
        similar = self.search_similar(current_query, n_results=n_results)
        
        # Also search for related terms to find more diverse context
        # For questions about names, also search for conversations mentioning names
        related_queries = []
        query_lower = current_query.lower()
        
        if "name" in query_lower or "who am i" in query_lower:
            related_queries.extend(["name", "my name", "I am", "call me"])
        elif "remember" in query_lower:
            # If asking about remembering, search for information that might have been shared
            related_queries.extend(["my", "I", "me"])
        
        # Search with related queries and combine results
        all_results = {}
        for result in similar:
            all_results[result['id']] = result
        
        for related_query in related_queries[:2]:  # Limit to avoid too many searches
            try:
                related_results = self.search_similar(related_query, n_results=n_results)
                for result in related_results:
                    if result['id'] not in all_results:
                        all_results[result['id']] = result
            except Exception as e:
                logger.debug(f"Error searching for related query '{related_query}': {e}")
        
        # Convert back to list and limit to n_results
        similar = list(all_results.values())[:n_results]
        
        if not similar:
            return ""
        
        context_parts = ["Previous similar conversations:"]
        for i, result in enumerate(similar, 1):
            metadata = result.get('metadata', {})
            user_input = metadata.get('user_input', '')
            assistant_response = metadata.get('assistant_response', '')
            context_parts.append(
                f"\n{i}. User: {user_input}\n   Assistant: {assistant_response}"
            )
        
        return "\n".join(context_parts)
    
    def add_tool_description(
        self,
        tool_name: str,
        description: str,
        input_schema: Optional[Dict] = None
    ) -> str:
        """
        Add a tool description to the vector database
        
        Args:
            tool_name: Name of the tool
            description: Description of the tool
            input_schema: Optional input schema
            
        Returns:
            The ID of the added document
        """
        document_text = f"Tool: {tool_name}\nDescription: {description}"
        
        if input_schema:
            document_text += f"\nSchema: {json.dumps(input_schema, indent=2)}"
        
        doc_id = f"tool_{tool_name}"
        
        # Update or add (ChromaDB will update if ID exists)
        self.collection.upsert(
            documents=[document_text],
            metadatas=[{
                "type": "tool",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[doc_id]
        )
        
        logger.info(f"Added/updated tool description: {tool_name}")
        return doc_id
    
    def search_tools(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant tools based on a query
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant tools
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"type": "tool"}
        )
        
        formatted_results = []
        if results and results.get('ids'):
            ids_list = results.get('ids', [[]])
            if ids_list and len(ids_list[0]) > 0:
                ids = ids_list[0]
                documents_list = results.get('documents', [[]])
                documents = documents_list[0] if documents_list and documents_list else []
                metadatas_list = results.get('metadatas', [[]])
                metadatas = metadatas_list[0] if metadatas_list and metadatas_list else []
                distances_list = results.get('distances')
                distances = distances_list[0] if distances_list and distances_list else []
                for i in range(len(ids)):
                    formatted_results.append({
                        "id": ids[i],
                        "document": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if distances and i < len(distances) else None
                    })
        
        return formatted_results
    
    def delete_conversation(self, doc_id: str) -> bool:
        """
        Delete a conversation from the vector database
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted conversation: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_directory
        }


# Global instance (can be initialized per application)
_vector_db_instance: Optional[VectorDBService] = None


def get_vector_db(persist_directory: str = "./chroma_db", collection_name: str = "conversations") -> VectorDBService:
    """
    Get or create the global vector database instance
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the collection to use
        
    Returns:
        VectorDBService instance
    """
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDBService(persist_directory, collection_name)
    return _vector_db_instance

