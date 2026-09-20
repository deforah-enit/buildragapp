"""
Streamlit Frontend for Hybrid RAG System
Provides document upload and chat interface
"""

import streamlit as st
import requests
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/ingest"
QUERY_ENDPOINT = f"{BACKEND_URL}/query"
STATUS_ENDPOINT = f"{BACKEND_URL}/status"
RESET_ENDPOINT = f"{BACKEND_URL}/reset"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .status-ready {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-waiting {
        color: #e74c3c;
        font-weight: bold;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #555;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85em;
        border-left: 3px solid #ffc107;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #ddd;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85em;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.ConnectionError:
        return False, None
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False, None

def get_session_status():
    """Get current session status"""
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return None

def upload_document(file):
    """Upload document to backend"""
    try:
        with st.spinner(f"📄 Processing '{file.name}'..."):
            files = {'file': file}
            # Increased timeout to 300 seconds (5 minutes) for entity extraction and graph creation
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=300)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return False, error_detail
    except requests.exceptions.Timeout:
        return False, "Request timed out. Document may be too large or backend is slow."
    except Exception as e:
        return False, f"Upload error: {str(e)}"

def query_document(question):
    """Send query to backend"""
    try:
        payload = {"question": question}
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return False, error_detail
    except requests.exceptions.Timeout:
        return False, "Request timed out. Try a simpler question."
    except Exception as e:
        return False, f"Query error: {str(e)}"

def reset_session():
    """Reset session"""
    try:
        response = requests.post(RESET_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, "Reset failed"
    except Exception as e:
        return False, str(e)

# ==================== SESSION STATE ====================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'document_ingested' not in st.session_state:
    st.session_state.document_ingested = False

if 'current_document' not in st.session_state:
    st.session_state.current_document = None

if 'backend_ready' not in st.session_state:
    st.session_state.backend_ready = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Settings & Info")
    
    # Backend health check
    backend_ready, health_info = check_backend_health()
    st.session_state.backend_ready = backend_ready
    
    if backend_ready:
        st.success("✓ Backend connected")
        if health_info:
            with st.expander("Connection Details"):
                st.write(f"**Neo4j:** {'✓ Connected' if health_info.get('neo4j_connected') else '✗ Not connected'}")
                st.write(f"**OpenAI:** {'✓ Connected' if health_info.get('openai_available') else '✗ Not available'}")
    else:
        st.error("✗ Backend not running")
        st.info("Please start the backend:\n```bash\ncd backend\npython -m uvicorn main:app --reload\n```")
    
    st.divider()
    
    # Document upload section
    st.markdown("### 📤 Upload Document")
    st.markdown(
        "Upload a new document to start a fresh chat session. "
        "Supported formats: PDF, DOCX, TXT"
    )
    
    uploaded_file = st.file_uploader(
        "Choose a file:",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True, key="ingest_btn"):
            if not backend_ready:
                st.error("Backend is not ready. Please start it first.")
            else:
                success, result = upload_document(uploaded_file)
                
                if success:
                    st.session_state.document_ingested = True
                    st.session_state.current_document = result['document_id']
                    st.session_state.chat_history = []  # Clear chat history
                    
                    st.success(f"✓ Document ingested successfully!")
                    with st.expander("Ingestion Details"):
                        st.write(f"**Document:** {result['document_id']}")
                        st.write(f"**Total Chunks:** {result['total_chunks']}")
                        st.write(f"**Vector Indexed:** {result['vector_chunks']}")
                        st.write(f"**Graph Relationships:** {result['graph_nodes']}")
                    st.rerun()
                else:
                    st.error(f"❌ Ingestion failed: {result}")
    
    st.divider()
    
    # Current session info
    if st.session_state.document_ingested:
        st.markdown("### 📋 Current Session")
        st.info(f"**Document:** {st.session_state.current_document}")
        st.write(f"**Messages:** {len(st.session_state.chat_history)}")
        
        if st.button("🔄 Upload Different Document", use_container_width=True):
            success, _ = reset_session()
            if success:
                st.session_state.document_ingested = False
                st.session_state.current_document = None
                st.session_state.chat_history = []
                st.success("Session reset. Ready for new document.")
                st.rerun()
    else:
        st.warning("📁 No document loaded")
        st.info("Upload a document to get started.")
    
    st.divider()
    
    # Help section
    with st.expander("❓ How it works"):
        st.markdown("""
        1. **Upload:** Choose and ingest a document
        2. **Process:** Document is analyzed by:
           - Vector DB (semantic similarity)
           - Knowledge Graph (entity relationships)
        3. **Query:** Ask questions about the document
        4. **Answer:** Get responses grounded in the document content
        """)

# ==================== MAIN CONTENT ====================
st.markdown('<h1 class="main-title">🤖 Hybrid RAG Chat System</h1>', unsafe_allow_html=True)

if not backend_ready:
    st.error(
        """
        ❌ **Backend is not running**
        
        Please start the backend server first:
        ```bash
        cd backend
        python -m uvicorn main:app --reload
        ```
        Then refresh this page.
        """
    )
    st.stop()

# Check if document is ingested
if not st.session_state.document_ingested:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📄 Getting Started
        
        1. **Upload a document** using the sidebar
        2. **Wait for processing** (vector embeddings + knowledge graph)
        3. **Start chatting** about the document content
        
        ### Supported Formats
        - 📑 PDF
        - 📝 DOCX (Word)
        - 🗂️ TXT, Markdown
        """)
    
    with col2:
        st.info("""
        ### ✨ Features
        
        **Hybrid Search:**
        - Vector similarity for semantic understanding
        - Knowledge graph for entity relationships
        - Combined scoring for better results
        
        **Smart Answers:**
        - Grounded in document content
        - Source citations included
        - Confidence scores for transparency
        """)
else:
    # Document is ingested - show chat interface
    st.success(f"📚 **Loaded:** {st.session_state.current_document}")
    
    # Display chat history
    st.markdown("### 💬 Conversation")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            html = f"""
            <div class="chat-message assistant">
                <strong>Assistant:</strong><br>
                {message['content']}
            """
            
            # Add sources if available
            if 'sources' in message and message['sources']:
                html += '<div class="source-box"><strong>Sources:</strong><br>'
                for i, source in enumerate(message['sources'], 1):
                    html += f"<div>[Source {i}] {source.get('snippet', 'N/A')[:100]}...</div>"
                html += '</div>'
            
            # Add confidence if available
            if 'confidence' in message:
                confidence_pct = int(message['confidence'] * 100)
                html += f'<div class="confidence-badge">Confidence: {confidence_pct}%</div>'
            
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("### 🔍 Ask a Question")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "What would you like to know?",
            label_visibility="collapsed",
            placeholder="Ask about the document...",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    # Process query
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Query backend
        with st.spinner("🤔 Thinking..."):
            success, result = query_document(user_input)
        
        if success:
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result.get('sources', []),
                "confidence": result.get('confidence', 0)
            })
            st.rerun()
        else:
            st.error(f"Error: {result}")

# ==================== FOOTER ====================
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 2em;">
    Hybrid RAG System | Powered by OpenAI, Neo4j, FAISS, and Claude
    </div>
    """,
    unsafe_allow_html=True
)