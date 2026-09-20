"""
FastAPI Backend for Hybrid RAG System
Handles document ingestion, vector search, graph search, and hybrid evaluation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import logging
from datetime import datetime
import signal
import sys

# Import our modules
from ingestion import DocumentProcessor
from vector_rag import VectorRAGPipeline
from graph_rag import GraphRAGPipeline
from hybrid_rag import HybridRAGEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup signal handler for graceful Ctrl+C interruption
def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n🛑 Interrupt signal received (Ctrl+C) - shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Initialize FastAPI
app = FastAPI(
    title="Hybrid RAG System API",
    description="Document ingestion and hybrid retrieval-augmented generation",
    version="1.0.0"
)

# Add CORS middleware for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================
# In production, use Redis or database; for now, in-memory is fine
class AppState:
    def __init__(self):
        self.current_doc_id = None
        self.document_processor = DocumentProcessor()
        self.vector_rag = VectorRAGPipeline()
        self.graph_rag = GraphRAGPipeline()
        self.hybrid_evaluator = HybridRAGEvaluator()
        self.chunks = []
        self.ingestion_complete = False

app_state = AppState()

# ==================== REQUEST/RESPONSE MODELS ====================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    session_id: str

class IngestionResponse(BaseModel):
    document_id: str
    total_chunks: int
    vector_chunks: int
    graph_nodes: int
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    openai_available: bool
    timestamp: str

# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and connectivity"""
    try:
        neo4j_ok = app_state.graph_rag.test_connection()
    except Exception as e:
        logger.warning(f"Neo4j connection test failed: {e}")
        neo4j_ok = False
    
    try:
        openai_ok = app_state.vector_rag.test_openai_connection()
    except Exception as e:
        logger.warning(f"OpenAI connection test failed: {e}")
        openai_ok = False
    
    return HealthResponse(
        status="healthy" if neo4j_ok and openai_ok else "degraded",
        neo4j_connected=neo4j_ok,
        openai_available=openai_ok,
        timestamp=datetime.now().isoformat()
    )

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document
    - Extracts text
    - Creates chunks
    - Generates embeddings
    - Builds knowledge graph
    - Complete session reset
    """
    try:
        # Reset previous session
        logger.info("Resetting session for new document ingestion...")
        app_state.vector_rag.reset()
        app_state.graph_rag.reset()
        app_state.chunks = []
        app_state.current_doc_id = None
        app_state.ingestion_complete = False
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing document: {file.filename}")
        
        # Step 1: Extract and chunk
        chunks = app_state.document_processor.process_document(temp_file_path)
        app_state.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("No text extracted from document")
        
        # Step 2: Vector pipeline - generate embeddings and store in FAISS
        logger.info("Processing vector pipeline...")
        vector_count = app_state.vector_rag.ingest_chunks(chunks)
        logger.info(f"Indexed {vector_count} chunks in vector DB")
        
        # Step 3: Graph pipeline - extract entities and build knowledge graph
        logger.info("Processing graph pipeline...")
        graph_count = app_state.graph_rag.ingest_chunks(chunks)
        logger.info(f"Created knowledge graph with {graph_count} relationships")
        
        # Mark as ready
        app_state.current_doc_id = file.filename
        app_state.ingestion_complete = True
        
        # Cleanup temp file
        os.remove(temp_file_path)
        os.rmdir(temp_dir)
        
        return IngestionResponse(
            document_id=app_state.current_doc_id,
            total_chunks=len(chunks),
            vector_chunks=vector_count,
            graph_nodes=graph_count,
            status="success",
            message=f"Document '{file.filename}' successfully ingested. Ready for queries."
        )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the ingested document using hybrid RAG
    1. Vector search (semantic similarity)
    2. Graph search (entity relationships)
    3. Hybrid evaluation and ranking
    4. LLM-based answer generation
    """
    try:
        if not app_state.ingestion_complete:
            raise ValueError("No document ingested yet. Please upload a document first.")
        
        logger.info(f"Processing query: {request.question}")
        
        # Step 1: Vector search
        logger.info("Executing vector search...")
        vector_results = app_state.vector_rag.search(request.question, top_k=5)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Step 2: Graph search
        logger.info("Executing graph search...")
        graph_results = app_state.graph_rag.search(request.question, top_k=5)
        logger.info(f"Graph search returned {len(graph_results)} results")
        
        # Step 3: Hybrid evaluation
        logger.info("Evaluating hybrid results...")
        final_answer, sources, confidence = app_state.hybrid_evaluator.evaluate(
            question=request.question,
            vector_results=vector_results,
            graph_results=graph_results,
            chunks=app_state.chunks,
            vector_weight=0.6,
            graph_weight=0.4
        )
        
        logger.info(f"Generated answer with confidence: {confidence:.2f}")
        
        return QueryResponse(
            answer=final_answer,
            sources=sources,
            confidence=confidence,
            session_id=app_state.current_doc_id
        )
    
    except ValueError as e:
        logger.warning(f"Query validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get current session status"""
    return {
        "document_ingested": app_state.ingestion_complete,
        "current_document": app_state.current_doc_id,
        "total_chunks": len(app_state.chunks),
        "ready_for_queries": app_state.ingestion_complete
    }

@app.post("/reset")
async def reset_session():
    """Manually reset the current session"""
    try:
        logger.info("Manual session reset requested")
        app_state.vector_rag.reset()
        app_state.graph_rag.reset()
        app_state.chunks = []
        app_state.current_doc_id = None
        app_state.ingestion_complete = False
        
        return {
            "status": "success",
            "message": "Session reset complete. Ready for new document upload."
        }
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Hybrid RAG Backend...")
    try:
        # Test Neo4j connection
        if app_state.graph_rag.test_connection():
            logger.info("✓ Neo4j connection established")
        else:
            logger.warning("⚠ Neo4j connection failed - check credentials")
        
        # Test OpenAI connection
        if app_state.vector_rag.test_openai_connection():
            logger.info("✓ OpenAI API connection established")
        else:
            logger.warning("⚠ OpenAI connection failed - check API key")
        
        logger.info("Backend startup complete")
    except Exception as e:
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )