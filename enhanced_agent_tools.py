# enhanced_agent_tools.py
# Enhanced agent that considers document importance and performs iterative searches

import logging
from typing import List, Dict, Any, Optional, Set
import re
import json
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core import Settings

logger = logging.getLogger(__name__)

class EnhancedDocumentAgentTools:
    """
    Enhanced tools that provide better document discovery for agents
    """
    
    @staticmethod
    def create_document_discovery_tool(
        document_metadata: Dict[str, 'DocumentMetadata'],
        vector_indexes: Dict[str, Any]
    ) -> FunctionTool:
        """
        Create a tool that helps the agent discover relevant documents
        beyond just summaries
        """
        def discover_documents(query: str) -> str:
            """
            Discover documents that might contain relevant information,
            including content buried deep in large documents.
            """
            logger.info(f"🔍 Document discovery called with query: {query}")
            try:
                # Analyze query to identify key concepts
                key_concepts = extract_key_concepts(query)
                
                # Score documents based on multiple factors
                doc_scores = {}
                for doc_id, metadata in document_metadata.items():
                    score = 0.0
                    
                    # 1. Summary relevance (existing approach)
                    summary_score = calculate_summary_relevance(
                        query, metadata.summary, key_concepts
                    )
                    
                    # 2. Document importance (prefer comprehensive sources)
                    importance_score = metadata.importance_score
                    
                    # 3. Quick sample search in large documents
                    if metadata.doc_type in ['book', 'pdf'] and metadata.chunk_count > 20:
                        # Do a quick vector search in this specific document
                        sample_score = quick_document_probe(
                            doc_id, query, vector_indexes.get(doc_id)
                        )
                    else:
                        sample_score = 0.0
                    
                    # Combined score with weights
                    score = (
                        summary_score * 0.4 +
                        importance_score * 0.3 +
                        sample_score * 0.3
                    )
                    
                    doc_scores[doc_id] = {
                        'score': score,
                        'summary_relevance': summary_score,
                        'importance': importance_score,
                        'content_relevance': sample_score,
                        'metadata': metadata
                    }
                
                # Format response for agent
                sorted_docs = sorted(
                    doc_scores.items(),
                    key=lambda x: x[1]['score'],
                    reverse=True
                )
                logger.info(f"📊 Discovery results:")
                for doc_id, info in sorted_docs[:5]:
                    logger.info(
                        f"  Doc {doc_id}: score={info['score']:.2f}, "
                        f"summary_rel={info['summary_relevance']:.2f}, "
                        f"importance={info['importance']:.2f}, "
                        f"content_rel={info['content_relevance']:.2f}"
                    )
                response = "Document discovery results:\n\n"
                for doc_id, info in sorted_docs[:10]:  # Top 10 documents
                    meta = info['metadata']
                    response += f"Document {doc_id} (Score: {info['score']:.2f}):\n"
                    response += f"  Type: {meta.doc_type}, Size: {meta.chunk_count} chunks\n"
                    response += f"  Summary: {meta.summary[:100]}...\n"
                    
                    if info['content_relevance'] > 0.5:
                        response += f"  ⚠️ HIGH RELEVANCE found in document content!\n"
                    
                    response += "\n"
                
                return response
                
            except Exception as e:
                logger.error(f"Document discovery error: {str(e)}")
                return f"Error during document discovery: {str(e)}"
        
        return FunctionTool.from_defaults(
            name="discover_relevant_documents",
            fn=discover_documents,
            description=(
                "Discover documents that might contain relevant information. "
                "This tool analyzes both document summaries AND samples content "
                "from large documents to find buried information."
            )
        )


def extract_key_concepts(query: str) -> List[str]:
    """Extract key concepts from query for better matching"""
    # Simple implementation - can be enhanced with NLP
    # Remove common words and extract key terms
    stop_words = {'what', 'how', 'when', 'where', 'why', 'is', 'are', 'the', 'a', 'an'}
    words = query.lower().split()
    concepts = [w for w in words if w not in stop_words and len(w) > 3]
    return concepts


def calculate_summary_relevance(query: str, summary: str, key_concepts: List[str]) -> float:
    """Calculate how relevant a summary is to the query"""
    if not summary:
        return 0.0
    
    query_lower = query.lower()
    summary_lower = summary.lower()
    
    # Direct query match
    if query_lower in summary_lower:
        return 1.0
    
    # Key concept matches
    matches = sum(1 for concept in key_concepts if concept in summary_lower)
    concept_score = matches / len(key_concepts) if key_concepts else 0
    
    return min(1.0, concept_score)


def quick_document_probe(doc_id: str, query: str, vector_index: Any) -> float:
    """Do a quick vector search in a specific document to check relevance"""
    if not vector_index:
        return 0.0
    
    try:
        # Quick search with just 3 results
        query_engine = vector_index.as_query_engine(
            similarity_top_k=3,
            response_mode="no_text"  # Just need scores, not text
        )
        
        response = query_engine.query(query)
        
        # Check if we got high-scoring results
        if hasattr(response, 'source_nodes') and response.source_nodes:
            # Average score of top results
            scores = [node.score for node in response.source_nodes if hasattr(node, 'score')]
            if scores:
                avg_score = sum(scores) / len(scores)
                return min(1.0, avg_score * 1.5)  # Boost to make it comparable
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error probing document {doc_id}: {str(e)}")
        return 0.0