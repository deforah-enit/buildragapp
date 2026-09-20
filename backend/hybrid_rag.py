"""
Hybrid RAG Evaluator Module
Merges vector and graph search results
Generates answers using OpenAI GPT-4o-mini API
"""

import logging
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

class HybridRAGEvaluator:
    """Evaluate hybrid search results and generate answers"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize evaluator
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file"
            )
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def _merge_and_deduplicate(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[Dict]:
        """
        Merge results from both pipelines and deduplicate
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            
        Returns:
            Merged and deduplicated results
        """
        merged = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result.get('chunk_id')
            if chunk_id not in merged:
                merged[chunk_id] = {
                    'text': result['text'],
                    'chunk_id': chunk_id,
                    'vector_score': result.get('score', 0),
                    'graph_score': 0,
                    'sources': [result.get('search_type', 'vector')]
                }
            else:
                merged[chunk_id]['vector_score'] = result.get('score', 0)
        
        # Add graph results
        for result in graph_results:
            chunk_id = result.get('chunk_id')
            if chunk_id not in merged:
                merged[chunk_id] = {
                    'text': result['text'],
                    'chunk_id': chunk_id,
                    'vector_score': 0,
                    'graph_score': result.get('score', 0),
                    'sources': [result.get('search_type', 'graph')]
                }
            else:
                merged[chunk_id]['graph_score'] = result.get('score', 0)
                if result.get('search_type', 'graph') not in merged[chunk_id]['sources']:
                    merged[chunk_id]['sources'].append(result.get('search_type', 'graph'))
        
        return list(merged.values())
    
    def _rank_results(
        self,
        merged_results: List[Dict],
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> List[Dict]:
        """
        Rank merged results using hybrid scoring
        
        Args:
            merged_results: Merged results from both pipelines
            vector_weight: Weight for vector score
            graph_weight: Weight for graph score
            
        Returns:
            Ranked results
        """
        for result in merged_results:
            # Combine scores
            vector_score = result.get('vector_score', 0)
            graph_score = result.get('graph_score', 0)
            
            # Hybrid score: weighted combination
            result['hybrid_score'] = (
                vector_weight * vector_score +
                graph_weight * graph_score
            )
            
            # Bonus for appearing in both pipelines
            if vector_score > 0 and graph_score > 0:
                result['hybrid_score'] *= 1.1  # 10% boost
        
        # Sort by hybrid score
        merged_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return merged_results
    
    def _build_context(self, ranked_results: List[Dict], top_k: int = 3) -> Tuple[str, List[Dict]]:
        """
        Build context string from top-ranked results
        
        Args:
            ranked_results: Ranked results
            top_k: Number of top results to include
            
        Returns:
            Tuple of (context_string, source_metadata)
        """
        context_parts = []
        sources = []
        
        for i, result in enumerate(ranked_results[:top_k]):
            source_info = {
                'chunk_id': result.get('chunk_id'),
                'score': result.get('hybrid_score', 0),
                'search_types': result.get('sources', []),
                'snippet': result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            }
            sources.append(source_info)
            
            context_parts.append(
                f"[Source {i+1}] (Score: {result.get('hybrid_score', 0):.2f})\n"
                f"{result['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        return context, sources
    
    def evaluate(
        self,
        question: str,
        vector_results: List[Dict],
        graph_results: List[Dict],
        chunks: List[Dict],
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> Tuple[str, List[Dict], float]:
        """
        Full evaluation pipeline: merge, rank, and generate answer
        
        Args:
            question: User question
            vector_results: Results from vector search
            graph_results: Results from graph search
            chunks: Original chunks (for reference)
            vector_weight: Weight for vector score
            graph_weight: Weight for graph score
            
        Returns:
            Tuple of (answer, sources, confidence)
        """
        logger.info("Starting hybrid evaluation...")
        
        # Step 1: Merge and deduplicate
        merged = self._merge_and_deduplicate(vector_results, graph_results)
        logger.info(f"Merged {len(merged)} unique results")
        
        # Step 2: Rank
        ranked = self._rank_results(merged, vector_weight, graph_weight)
        logger.info(f"Ranked {len(ranked)} results")
        
        # Step 3: Build context
        context, sources = self._build_context(ranked, top_k=3)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence_score(ranked, vector_results, graph_results)
        
        # Step 4: Generate answer using GPT-4o-mini
        logger.info("Generating answer with GPT-4o-mini...")
        answer = self._generate_answer_gpt(question, context)
        
        logger.info(f"Answer generated with confidence: {confidence:.2%}")
        
        return answer, sources, confidence
    
    def _calculate_confidence_score(
        self,
        ranked_results: List[Dict],
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> float:
        """
        Calculate confidence score based on multiple factors:
        1. Top result hybrid score (search relevance)
        2. Number of supporting sources (more = better)
        3. Agreement between vector and graph results (both found = better)
        
        Returns: Confidence as float (0.0 to 1.0), suitable for percentage display
        """
        if not ranked_results:
            return 0.0
        
        # Factor 1: Top result score (typically 0.2-0.8 from FAISS)
        top_score = ranked_results[0]['hybrid_score']
        
        # Factor 2: Number of supporting sources
        # Using top 3 results: 1 source = 0.33, 2 sources = 0.67, 3 sources = 1.0
        num_sources = min(len(ranked_results[:3]), 3)
        source_factor = num_sources / 3.0
        
        # Factor 3: Agreement between vector and graph
        # If top result came from both vector AND graph search, higher confidence
        top_chunk_id = ranked_results[0].get('chunk_id')
        vector_found = any(r.get('chunk_id') == top_chunk_id for r in vector_results)
        graph_found = any(r.get('chunk_id') == top_chunk_id for r in graph_results)
        
        if vector_found and graph_found:
            agreement_factor = 1.0  # Both found it - very confident
        elif vector_found or graph_found:
            agreement_factor = 0.85  # One found it - reasonably confident
        else:
            agreement_factor = 0.7  # Neither found it directly - lower confidence
        
        # Combine factors: 45% top_score, 35% sources, 20% agreement
        combined = (
            0.45 * top_score +
            0.35 * source_factor +
            0.20 * agreement_factor
        )
        
        # Normalize: FAISS scores are typically 0.2-0.8, so scale up to 0-1 range
        # This ensures better answers get 70-95% confidence
        confidence = min(combined * 1.2, 1.0)
        
        logger.debug(
            f"Confidence: top_score={top_score:.3f}, "
            f"sources={source_factor:.3f}, "
            f"agreement={agreement_factor:.3f}, "
            f"final={confidence:.3f} ({confidence:.1%})"
        )
        
        return confidence
    
    def _generate_answer_gpt(self, question: str, context: str) -> str:
        """
        Generate answer using OpenAI GPT-4o-mini API
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        system_prompt = """You are a helpful assistant that answers questions based on provided documents.
You have access to relevant excerpts from a document. Answer the question accurately and completely using ONLY the information provided in the context.

IMPORTANT: Pay special attention to sections marked with [ENTITY: ...] tags. These contain extracted entity information including:
- Entity names
- QUANTITY: The recommended serving size or measurement
- DESCRIPTION: What the entity is
- CATEGORY: The category it belongs to

Always use these entity details when answering questions about specific items.

If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which source you're using for your answer.
Be concise but complete in your response.
If you don't know something, say "I don't have this information in the provided document." """
        
        user_prompt = f"""Question: {question}

Context from document:
{context}

Please answer the question based on the provided context."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response.choices[0].message.content
            return answer
        
        except Exception as e:
            logger.error(f"OpenAI GPT API error: {e}")
            raise