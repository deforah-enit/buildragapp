"""
Document Ingestion Module
Handles file parsing, text extraction, cleaning, and chunking
Supports PDF, DOCX, TXT and other common formats
"""

import os
import logging
import re
from typing import List, Dict
from pathlib import Path

# PDF handling
try:
    import pypdf
except ImportError:
    pypdf = None

# DOCX handling
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# File type detection
import mimetypes

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents and create chunks for RAG"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 50
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target tokens per chunk (~4 chars per token)
            chunk_overlap: Tokens of overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.chars_per_token = 4  # Approximate
        
    def process_document(self, file_path: str) -> List[Dict]:
        """
        Main entry point: process any supported document type
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of chunks with metadata
        """
        logger.info(f"Processing document: {file_path}")
        
        # Detect file type
        file_ext = Path(file_path).suffix.lower()
        
        # Extract text based on file type
        if file_ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif file_ext == '.docx':
            text = self._extract_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            text = self._extract_text(file_path)
        else:
            # Try to read as text
            logger.warning(f"Unsupported file type {file_ext}, attempting text extraction")
            text = self._extract_text(file_path)
        
        if not text or not text.strip():
            raise ValueError(f"No text could be extracted from {file_path}")
        
        logger.info(f"Extracted {len(text)} characters from document")
        
        # Clean text
        text = self._clean_text(text)
        logger.info(f"Cleaned text: {len(text)} characters")
        
        # Create chunks
        chunks = self._chunk_text(text, file_path)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        if pypdf is None:
            raise ImportError("pypdf not installed. Install with: pip install pypdf")
        
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            logger.info(f"Extracted text from PDF: {len(reader.pages)} pages")
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Failed to extract PDF: {e}")
        
        return text
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        if DocxDocument is None:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        
        text = ""
        try:
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            logger.info(f"Extracted text from DOCX: {len(doc.paragraphs)} paragraphs")
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise ValueError(f"Failed to extract DOCX: {e}")
        
        return text
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Extracted text file: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise ValueError(f"Failed to extract text: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters but keep newlines
        text = ''.join(ch for ch in text if ch.isprintable() or ch == '\n')
        
        # Fix common OCR issues
        text = re.sub(r'(?<=[a-z])\.(?=[A-Z])', '. ', text)  # Fix missing spaces after periods
        
        # Remove lines that are just numbers/symbols (likely page numbers)
        lines = text.split('\n')
        lines = [line for line in lines if not re.match(r'^[\d\-\s]+$', line)]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _chunk_text(
        self,
        text: str,
        file_path: str,
        chunk_size_chars: int = None
    ) -> List[Dict]:
        """
        Split text into overlapping chunks - FAST and SAFE version
        
        Args:
            text: Full document text
            file_path: Original file path (for metadata)
            chunk_size_chars: Size in characters (default: chunk_size * chars_per_token)
            
        Returns:
            List of chunks with metadata
        """
        if chunk_size_chars is None:
            chunk_size_chars = self.chunk_size * self.chars_per_token
        
        overlap_chars = self.chunk_overlap * self.chars_per_token
        
        chunks = []
        chunk_id = 0
        
        logger.info(f"Starting chunking: text_len={len(text)}, chunk_size={chunk_size_chars}, overlap={overlap_chars}")
        
        # SAFE APPROACH: Use start/end indices only, no fancy boundary detection
        position = 0
        MAX_CHUNKS = 10000  # Safety limit - should never reach this for normal documents
        
        while position < len(text):
            # Safety check - prevent infinite loops
            if chunk_id >= MAX_CHUNKS:
                logger.error(f"🚨 SAFETY LIMIT REACHED: {chunk_id} chunks created, stopping!")
                logger.error(f"Position: {position}/{len(text)}")
                raise ValueError(f"Chunking safety limit exceeded ({MAX_CHUNKS} chunks). Text may be malformed.")
            
            # Calculate chunk boundaries
            chunk_start = position
            chunk_end = min(position + chunk_size_chars, len(text))
            
            # Extract chunk
            chunk_text = text[chunk_start:chunk_end].strip()
            
            # Skip tiny chunks
            if len(chunk_text) >= self.min_chunk_size:
                chunk = {
                    'id': f"{Path(file_path).stem}_chunk_{chunk_id}",
                    'text': chunk_text,
                    'start_char': chunk_start,
                    'end_char': chunk_end,
                    'source_file': os.path.basename(file_path),
                    'chunk_index': chunk_id,
                    'token_count': len(chunk_text) // self.chars_per_token
                }
                chunks.append(chunk)
                chunk_id += 1
                logger.info(f"Created chunk {chunk_id}: {len(chunk_text)} chars, position {chunk_start}-{chunk_end}")
            
            # Move position forward for next chunk
            # Key: always move forward by (chunk_size - overlap)
            step = chunk_size_chars - overlap_chars
            
            # Safety: if step is 0 or negative, force forward progress
            if step <= 0:
                logger.warning(f"⚠️ Overlap ({overlap_chars}) >= chunk_size ({chunk_size_chars}), forcing step to chunk_size")
                step = chunk_size_chars
            
            position += step
            
            # Final safety: if we're at the end, break
            if chunk_end >= len(text):
                logger.info(f"Reached end of text at position {chunk_end}")
                break
        
        if not chunks:
            raise ValueError("No valid chunks created from text")
        
        avg_size = sum(len(c['text']) for c in chunks) // len(chunks) if chunks else 0
        logger.info(f"✅ Chunking complete: {len(chunks)} chunks, avg size: {avg_size} chars")
        
        return chunks