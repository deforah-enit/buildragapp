"""
Vector RAG Module
Handles embedding generation and FAISS-based semantic search
Uses OpenAI's embedding API
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

# FAISS for vector search
try:
    import faiss
except ImportError:
    faiss = None

# OpenAI for embeddings
import openai
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

class VectorRAGPipeline:
    """Handle vector embeddings and semantic search using FAISS"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536
    ):
        """
        Initialize vector RAG pipeline
        
        Args:
            embedding_model: OpenAI embedding model
            embedding_dim: Dimension of embeddings (1536 for text-embedding-3-small)
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.client = None
        self.faiss_index = None
        self.chunk_texts = []  # Store original texts
        self.chunk_ids = []    # Store chunk IDs for reference
        self.embeddings = []   # Store embeddings
        
        self._initialize_openai()
        self._initialize_faiss()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file"
            )
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        try:
            if faiss is None:
                raise ImportError("faiss-cpu not installed")
            
            # Create a simple flat index
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("FAISS index initialized")
        except Exception as e:
            logger.error(f"FAISS initialization error: {e}")
            raise
    
    def test_openai_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            # Try a small embedding
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input="test"
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Truncate to avoid token limits
            text = text[:8000]
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    def ingest_chunks(self, chunks: List[Dict]) -> int:
        """
        Ingest chunks: generate embeddings and add to FAISS index
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            Number of chunks indexed
        """
        logger.info(f"Ingesting {len(chunks)} chunks into vector DB...")
        
        count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = self._embed_text(chunk['text'])
                
                # Add to FAISS index
                embedding_array = np.array([embedding], dtype=np.float32)
                self.faiss_index.add(embedding_array)
                
                # Store metadata
                self.chunk_texts.append(chunk['text'])
                self.chunk_ids.append(chunk['id'])
                self.embeddings.append(embedding)
                
                count += 1
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.get('id', i)}: {e}")
                # Continue with next chunk
                continue
        
        logger.info(f"Successfully indexed {count}/{len(chunks)} chunks")
        return count
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search using FAISS
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        if not self.chunk_texts:
            logger.warning("Vector DB is empty - no chunks indexed")
            return []
        
        try:
            # Embed the query
            query_embedding = self._embed_text(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_array, min(top_k, len(self.chunk_texts)))
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0:  # Valid index
                    # Convert L2 distance to similarity score (0-1)
                    # Lower distance = higher similarity
                    similarity = 1 / (1 + distance)
                    
                    results.append({
                        'text': self.chunk_texts[idx],
                        'chunk_id': self.chunk_ids[idx],
                        'score': float(similarity),
                        'distance': float(distance),
                        'search_type': 'vector'
                    })
            
            logger.info(f"Vector search found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def reset(self):
        """Reset vector DB for new session"""
        logger.info("Resetting vector DB...")
        self._initialize_faiss()
        self.chunk_texts = []
        self.chunk_ids = []
        self.embeddings = []
        logger.info("Vector DB reset complete")