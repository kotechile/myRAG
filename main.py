# main.py - Clean RAG Application with Fixed Agent Support

import os
import time
import logging
import threading
import gc
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from werkzeug.utils import secure_filename

# Flask imports
from flask import Flask, request, jsonify

# Environment and configuration 
from dotenv import load_dotenv, find_dotenv

# Supabase imports
from supabase import create_client
import vecs

# LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_parse import LlamaParse

# Your existing imports
from optimized_chunking import OptimizedDocumentChunker
from optimized_embedding import EmbeddingPipeline, BatchDocumentProcessor
from llm_provider import LLMProviderFactory
import uuid
import db_config  # Added for API key retrieval
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

# API keys and configurations
# Try to get Llama Cloud key from DB first, then env
LLAMA_PARSE_API_KEY = db_config.get_api_key("llama_cloud") or os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMA_PARSE_API_KEY:
    logger.warning("⚠️ LLAMA_CLOUD_API_KEY not found in Supabase or environment variables. LlamaParse may fail.")
else:
    logger.info("✅ LLAMA_CLOUD_API_KEY loaded successfully")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DATABASE_PASSWORD = os.getenv("SUPABASE_DATABASE_PASSWORD")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database configuration
DB_CONNECTION = f"postgresql://postgres.dgcsqiaciyqvprtpopxg:{SUPABASE_DATABASE_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
DEFAULT_LLM_PROVIDER = "deepseek"

# Global storage
active_jobs = {}
_engines = {}
# Use more conservative thread pool to prevent memory issues
background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag_background")

try:
    from knowledge_gap_http_supabase import integrate_knowledge_gap_filler_http
    KNOWLEDGE_GAP_AVAILABLE = True
    logger.info("✅ HTTP-based Knowledge Gap Filler available")
except ImportError:
    KNOWLEDGE_GAP_AVAILABLE = False
    logger.info("⚠️ Knowledge Gap Filler not available")

# Try to import optimization
try:
    from embedding_optimizer import OptimizedOpenAIEmbedding, CombinedOptimizer, create_128d_int8_optimizer
    OPTIMIZATION_AVAILABLE = True
    logger.info("✅ Embedding optimization module available")
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logger.info("⚠️ Embedding optimization not available, using standard embeddings")

# Try to import agents
try:
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import QueryEngineTool, FunctionTool
    AGENTS_AVAILABLE = True
    logger.info("✅ LlamaIndex agents available")
except ImportError as e:
    logger.warning(f"⚠️ LlamaIndex agents not available: {e}")
    AGENTS_AVAILABLE = False
jobs: Dict[str, Dict[str, Any]] = {}
job_lock = threading.Lock()  # Thread-safe access to jobs dictionary


def cleanup_old_jobs(max_age_seconds: int = 3600):
    """Remove completed jobs older than max_age_seconds."""
    current_time = time.time()
    with job_lock:
        jobs_to_remove = []
        for job_id, job_info in jobs.items():
            if job_info.get('status') in ['completed', 'failed']:
                # Check if job has a completion time
                completion_time = job_info.get('completion_time', current_time)
                if current_time - completion_time > max_age_seconds:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del jobs[job_id]
            logger.info(f"Cleaned up old job: {job_id}")


class JobStatus:
    """Enhanced job status tracking."""
    def __init__(self, job_id):
        self.job_id = job_id
        self.status = "starting"
        self.progress = 0
        self.result = None
        self.error = None
        self.steps = []
        self.current_step = ""
        self.detailed_status = {
            'phase': 'initialization',
            'documents_being_searched': [],
            'chunks_processed': 0,
            'search_strategy': '',
            'llm_provider': '',
            'embedding_dimension': 0,
            'current_document': '',
            'search_iterations': 0,
            'max_iterations': 0,
            'results_quality': 'unknown'
        }
        
    def update_status(self, status, progress=None, step=None, **detailed_kwargs):
        self.status = status
        if progress is not None:
            self.progress = progress
        if step:
            self.current_step = step
            self.steps.append(f"{time.strftime('%H:%M:%S')} - {step}")
            
        # Update detailed status
        for key, value in detailed_kwargs.items():
            if key in self.detailed_status:
                self.detailed_status[key] = value

class EmbeddingModelManager:
    """Manages embedding model with dynamic dimension detection."""
    
    def __init__(self):
        self._model = None
        self._dimension = None
        self._fitted = False
        
    def get_model(self):
        """Get the embedding model, creating it if necessary."""
        if self._model is None:
            self._model = self._create_embedding_model()
        return self._model
    
    def get_dimension(self):
        """Get the embedding dimension, detecting it if necessary."""
        if self._dimension is None:
            self._dimension = self._detect_dimension()
        return self._dimension
    
    def _create_embedding_model(self):
        """Create the appropriate embedding model."""
        use_optimization = os.getenv("USE_EMBEDDING_OPTIMIZATION", "false").lower() == "true"
        
        if use_optimization and OPTIMIZATION_AVAILABLE:
            logger.info("🔧 Creating optimized embedding model...")
            try:
                optimizer_path = "optimizer_128d_int8.joblib"
                if os.path.exists(optimizer_path):
                    optimizer = CombinedOptimizer.load(optimizer_path)
                    embed_model = OptimizedOpenAIEmbedding(
                        model="text-embedding-3-small",
                        optimizer=optimizer,
                        embed_batch_size=10
                    )
                    self._fitted = optimizer.fitted
                    logger.info(f"✅ Loaded optimizer: {optimizer.target_dim}D + {optimizer.quantization}")
                    return embed_model
                else:
                    optimizer = create_128d_int8_optimizer()
                    embed_model = OptimizedOpenAIEmbedding(
                        model="text-embedding-3-small",
                        optimizer=optimizer,
                        embed_batch_size=10
                    )
                    self._fitted = False
                    return embed_model
            except Exception as e:
                logger.warning(f"⚠️ Optimization setup failed: {e}, falling back to standard embeddings")
        
        # Fallback to standard embeddings
        logger.info("📐 Using standard OpenAI embeddings (1536D)")
        self._fitted = True
        return OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=10)
    
    def _detect_dimension(self):
        """Detect the embedding dimension dynamically."""
        try:
            model = self.get_model()
            
            if hasattr(model, 'optimizer') and model.optimizer and model.optimizer.fitted:
                dim = model.optimizer.target_dim
                logger.info(f"📏 Detected dimension from fitted optimizer: {dim}")
                return dim
            
            elif hasattr(model, 'optimizer') and model.optimizer and not model.optimizer.fitted:
                logger.info("🔧 Optimizer not fitted, using standard dimension temporarily")
                return 1536
            
            logger.info("📏 Testing embedding to determine dimension...")
            test_embedding = model.get_text_embedding("test")
            dim = len(test_embedding)
            logger.info(f"📏 Determined dimension by testing: {dim}")
            return dim
            
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            use_optimization = os.getenv("USE_EMBEDDING_OPTIMIZATION", "false").lower() == "true"
            return 128 if use_optimization and OPTIMIZATION_AVAILABLE else 1536
    
    def ensure_optimizer_fitted(self):
        """Ensure the optimizer is fitted if using optimization."""
        model = self.get_model()
        
        if hasattr(model, 'optimizer') and model.optimizer and not model.optimizer.fitted:
            logger.info("🔧 Fitting optimizer with sample texts...")
            
            sample_texts = [
                "Real estate investment and property management strategies",
                "Financial planning and portfolio optimization techniques", 
                "Machine learning algorithms and data science methods",
                "Natural language processing and text analysis tools",
                "Business development and market research approaches"
            ] * 52  # Create 260 samples
            
            try:
                model.fit_optimizer(sample_texts)
                self._fitted = True
                self._dimension = None  # Reset dimension to be re-detected
                logger.info("✅ Optimizer fitted successfully")
                
                # Save the fitted optimizer
                optimizer_path = "optimizer_128d_int8.joblib"
                model.optimizer.save(optimizer_path)
                logger.info(f"💾 Fitted optimizer saved to {optimizer_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to fit optimizer: {e}")
                raise

# Initialize global embedding manager
embedding_manager = EmbeddingModelManager()

# Set the embedding model in Settings
embed_model = embedding_manager.get_model()
Settings.embed_model = embed_model

# Initialize Supabase client
# Use the client already initialized in db_config to avoid duplicate connections and key errors
supabase = db_config.supabase

if not supabase:
    logger.error("❌ Failed to initialize Supabase client (from db_config)")
    # We could raise here, but let's allow it to continue and fail gracefully in endpoints if needed
else:
    logger.info("✅ Supabase client initialized successfully (from db_config)")

# Initialize LlamaParse client
parser = LlamaParse(
    api_key=LLAMA_PARSE_API_KEY,
    result_type="markdown",
    num_workers=4,
)

# Initialize default LLM from DB config
try:
    llm_config = db_config.get_llm_config()
    if llm_config:
        # Pass all config fields to the factory (llm_config already contains 'provider')
        default_llm = LLMProviderFactory.get_llm_instance(**llm_config)
        Settings.llm = default_llm
        logger.info(f"✅ Initialized {llm_config.get('provider')} ({llm_config.get('model')}) as default LLM from DB")
    else:
        logger.warning("No default LLM config found in DB, attempting fallback...")
        # Fallback to env-based initialization if DB config missing
        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        logger.info("Initialized fallback OpenAI LLM")
except Exception as e:
    logger.error(f"Error initializing default LLM: {str(e)}")
    # Last resort fallback to avoid crash
    try:
        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    except:
        pass

# Initialize components
document_chunker = OptimizedDocumentChunker(embed_model=embed_model)
embedding_pipeline = EmbeddingPipeline(embed_model=embed_model, db_connection=DB_CONNECTION)
document_processor = BatchDocumentProcessor(
    chunker=document_chunker,
    embedding_pipeline=embedding_pipeline,
    supabase_client=supabase
)

def create_vector_store_with_dynamic_dimension(collection_name: str):
    """Create vector store with dynamic dimension detection."""
    embedding_manager.ensure_optimizer_fitted()
    dimension = embedding_manager.get_dimension()
    logger.info(f"📏 Creating vector store '{collection_name}' with dimension: {dimension}")
    
    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION,
        collection_name=collection_name,
        dimension=dimension
    )
    
    return vector_store

class SimpleRAGEngine:
    """Clean, simple RAG engine with fixed agent support."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        # Use dynamic dimension detection
        self.vector_store = create_vector_store_with_dynamic_dimension(collection_name)
        
        # Create index from existing vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Load document metadata
        self.document_metadata = self._load_document_metadata()


    def query_hybrid_enhanced(self, query: str, top_k: int = 10, llm_provider: str = DEFAULT_LLM_PROVIDER, balance_emphasis: str = None) -> Dict[str, Any]:
        """
        Enhanced hybrid query using similarity search with importance weighting.
        Supports configurable balance between comprehensive sources and recent content.
        """
        start_time = time.time()
        
        # Auto-detect balance emphasis if not provided
        if balance_emphasis is None:
            balance_emphasis = self._detect_query_type(query)
        
        logger.info(f"🔍 Hybrid enhanced query: {query} (balance: {balance_emphasis})")
        
        try:
            # Get LLM
            llm_adapter = LLMProviderFactory.get_adapter(llm_provider)
            llm_instance = llm_adapter.get_llm_instance()
            
            # Perform similarity search with more results to allow for reranking
            retriever = self.vector_index.as_retriever(similarity_top_k=top_k * 2)
            
            def retrieve_with_timeout():
                return retriever.retrieve(query)
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(retrieve_with_timeout)
                try:
                    nodes = future.result(timeout=30)
                except FutureTimeoutError:
                    return {"status": "error", "error": "Vector retrieval timed out"}
            
            if not nodes:
                return {
                    "status": "success",
                    "query": query,
                    "response": "No relevant information found for your query.",
                    "method": "hybrid_enhanced",
                    "searchQuality": "basic",
                    "time_seconds": round(time.time() - start_time, 2),
                    "chunks_used": 0,
                    "documents_searched": 0,
                    "documents_used": [],
                    "source_attribution": [],
                    "enhancement_info": {
                        "importance_weighting_applied": False,
                        "initial_results": 0,
                        "after_reranking": 0,
                        "document_scores": {}
                    },
                    "llm_provider": llm_provider,
                    "embedding_dimension": embedding_manager.get_dimension()
                }
            
            # Apply importance weighting to rerank results
            weighted_nodes = []
            doc_scores = {}  # Track best score per document
            
            for node in nodes:
                doc_id = node.metadata.get('docid', '')
                if doc_id:
                    # Get document importance weight with balance emphasis
                    doc_metadata = self.document_metadata.get(str(doc_id), {})
                    weight = self._calculate_importance_weight(doc_metadata, balance_emphasis)
                    
                    # Apply weight to similarity score
                    boosted_score = node.score * weight
                    
                    # Track best score per document
                    if doc_id not in doc_scores or boosted_score > doc_scores[doc_id]['score']:
                        doc_scores[doc_id] = {
                            'score': boosted_score,
                            'weight': weight,
                            'original_score': node.score
                        }
                    
                    # Update node score
                    node.score = boosted_score
                    weighted_nodes.append(node)
            
            # Sort by boosted scores and take top K
            weighted_nodes.sort(key=lambda x: x.score, reverse=True)
            selected_nodes = weighted_nodes[:top_k]
            
            # Track document usage
            document_usage = {}
            total_chars = 0
            
            for node in selected_nodes:
                doc_id = node.metadata.get('docid', 'unknown')
                if doc_id != 'unknown':
                    if doc_id not in document_usage:
                        document_usage[doc_id] = {
                            'chunks': 0,
                            'characters': 0,
                            'weight': doc_scores.get(doc_id, {}).get('weight', 1.0)
                        }
                    document_usage[doc_id]['chunks'] += 1
                    document_usage[doc_id]['characters'] += len(node.text)
                    total_chars += len(node.text)
            
            # Create context from selected nodes
            context_parts = []
            for node in selected_nodes:
                text = node.text
                if len(text) > 800:  # Limit individual chunk size
                    text = text[:800] + "..."
                context_parts.append(text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate response
            prompt = f"""Answer the following question based on the provided context. 
            The context includes information from multiple documents, with preference given to comprehensive sources.
            
            QUESTION: {query}
            
            CONTEXT:
            {context}
            
            Provide a detailed and helpful answer based on the information above.
            
            ANSWER:"""
            
            def generate_response():
                return llm_instance.complete(prompt)
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_response)
                try:
                    response = future.result(timeout=60)
                except FutureTimeoutError:
                    return {"status": "error", "error": "LLM response generation timed out"}
            
            # Create document tracking with titles
            documents_used = []
            source_attribution = []
            
            for doc_id, usage_info in document_usage.items():
                # Get document title and author from metadata
                doc_metadata = self.document_metadata.get(str(doc_id), {})
                
                # If metadata not found or has default values, try direct lookup
                if not doc_metadata or doc_metadata.get('title') == f'Document {doc_id}':
                    logger.warning(f"HYBRID_ENHANCED - Metadata not found for document {doc_id}, attempting direct lookup...")
                    logger.warning(f"HYBRID_ENHANCED - Current collection_name: {self.collection_name}")
                    logger.warning(f"HYBRID_ENHANCED - Available metadata keys: {list(self.document_metadata.keys())}")
                    try:
                        # Use HTTPSupabaseClient for direct lookup to avoid compatibility issues
                        from knowledge_gap_http_supabase import HTTPSupabaseClient
                        http_supabase = HTTPSupabaseClient()
                        direct_response = http_supabase.table("lindex_documents").select(
                            "id, title, author, url, publish_date, in_vector_store, collectionId"
                        ).eq("id", doc_id).execute()
                        
                        if direct_response.data:
                            direct_doc = direct_response.data[0]
                            doc_title = direct_doc.get('title', f'Document {doc_id}')
                            doc_author = direct_doc.get('author', 'Unknown Author')
                            doc_url = direct_doc.get('url', '')
                            doc_publish_date = direct_doc.get('publish_date', '')
                            collection_id = direct_doc.get('collectionId')
                            logger.info(f"HYBRID_ENHANCED - Direct lookup SUCCESS for doc {doc_id}: title='{doc_title}', author='{doc_author}', url='{doc_url}', publish_date='{doc_publish_date}', collectionId={collection_id}")
                        else:
                            doc_title = f'Document {doc_id}'
                            doc_author = 'Unknown Author'
                            doc_url = ''
                            doc_publish_date = ''
                            logger.warning(f"HYBRID_ENHANCED - Document {doc_id} not found in database")
                    except Exception as e:
                        logger.error(f"HYBRID_ENHANCED - Error in direct lookup for document {doc_id}: {e}")
                        doc_title = f'Document {doc_id}'
                        doc_author = 'Unknown Author'
                        doc_url = ''
                        doc_publish_date = ''
                else:
                    doc_title = doc_metadata.get('title', f'Document {doc_id}')
                    doc_author = doc_metadata.get('author', 'Unknown Author')
                    doc_url = doc_metadata.get('url', '')
                    doc_publish_date = doc_metadata.get('publish_date', '')
                    logger.info(f"HYBRID_ENHANCED - Using cached metadata for doc {doc_id}: title='{doc_title}', author='{doc_author}', url='{doc_url}', publish_date='{doc_publish_date}'")
                
                # Debug: Log what we're getting for each document
                logger.info(f"HYBRID_ENHANCED - QUERY DEBUG - Document {doc_id}: metadata={doc_metadata}, title='{doc_title}', author='{doc_author}'")
                
                documents_used.append({
                    'document_id': doc_id,
                    'title': doc_title,
                    'author': doc_author,
                    'url': doc_url,
                    'publish_date': doc_publish_date,
                    'chunks_contributed': usage_info['chunks'],
                    'characters_retrieved': usage_info['characters'],
                    'importance_weight': usage_info['weight'],
                    'status': 'used'
                })
                
                source_attribution.append(f"{doc_title} by {doc_author}: {usage_info['chunks']} chunks (weight: {usage_info['weight']:.1f})")
            
            documents_used.sort(key=lambda x: x['chunks_contributed'], reverse=True)
            
            total_time = time.time() - start_time
            
            # Determine search quality based on results
            chunks_used = len(selected_nodes)
            if chunks_used >= 15 and len(document_usage) >= 3:
                search_quality = "premium"
            elif chunks_used >= 8 and len(document_usage) >= 2:
                search_quality = "advanced"
            else:
                search_quality = "basic"
            
            return {
                "status": "success",
                "query": query,
                "response": str(response),
                "method": "hybrid_enhanced",
                "searchQuality": search_quality,  # NOW PROPERLY CALCULATED
                "time_seconds": round(total_time, 2),
                "chunks_used": chunks_used,
                "documents_searched": len(document_usage),
                "documents_used": documents_used,
                "source_attribution": source_attribution,
                "enhancement_info": {
                    "importance_weighting_applied": True,
                    "initial_results": len(nodes),
                    "after_reranking": len(selected_nodes),
                    "document_scores": doc_scores
                },
                "llm_provider": llm_provider,
                "embedding_dimension": embedding_manager.get_dimension()
            }
                    
        except Exception as e:
            logger.error(f"❌ Hybrid enhanced query error: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "searchQuality": "basic"
            }
    # Also update the _load_document_metadata method to include titles
    def _load_document_metadata(self) -> Dict[str, Dict]:
        """Load document metadata for document selection (enhanced with titles)."""
        try:
            # Use HTTP client directly to avoid SDK compatibility issues
            from knowledge_gap_http_supabase import HTTPSupabaseClient
            http_supabase = HTTPSupabaseClient()
            response = http_supabase.table("lindex_documents").select(
                "id, summary_short, summary_medium, doc_size, chunk_count, source_type, title, author, url, publish_date, in_vector_store, collectionId"
            ).eq("in_vector_store", True).execute()
            
            logger.info(f"METADATA_LOAD - Found {len(response.data)} documents in vector store across all collections (via HTTP)")
            logger.info(f"METADATA_LOAD - Target collection_name: {self.collection_name}")
            
            # Get collection mapping using HTTP client
            collections_response = http_supabase.table("lindex_collections").select("id, name").execute()
            collection_map = {str(col["id"]): col["name"] for col in collections_response.data}
            logger.info(f"Available collections: {collection_map}")
            
            # Filter documents that belong to collections that match our collection name
            # or are in the same collection as documents we're interested in
            target_collection_id = None
            
            # First, try to find our target collection
            for col in collections_response.data:
                if col["name"] == self.collection_name:
                    target_collection_id = col["id"]
                    break
            
            if target_collection_id:
                logger.info(f"METADATA_LOAD - Loading metadata for collection: {self.collection_name} (ID: {target_collection_id})")
                # Filter documents by target collection
                filtered_docs = [doc for doc in response.data if doc.get("collectionId") == target_collection_id]
                logger.info(f"METADATA_LOAD - Filtered to {len(filtered_docs)} documents for collection {target_collection_id}")
            else:
                logger.warning(f"METADATA_LOAD - Collection '{self.collection_name}' not found. Loading all documents in vector store.")
                filtered_docs = response.data
                logger.info(f"METADATA_LOAD - Using all {len(filtered_docs)} documents")
            
            logger.info(f"METADATA_LOAD - Found {len(filtered_docs)} documents for processing")
            
            # Debug: Check if document 208 is in the results
            doc_208_found = False
            for doc in filtered_docs:
                if str(doc["id"]) == "208":
                    doc_208_found = True
                    collection_name = collection_map.get(str(doc.get("collectionId")), "Unknown")
                    logger.info(f"Document 208 found: title='{doc.get('title')}', author='{doc.get('author')}', collection='{collection_name}' (ID: {doc.get('collectionId')})")
                    break
            
            if not doc_208_found:
                logger.warning("Document 208 not found in filtered results. Checking all documents...")
                # Check if document 208 exists at all
                doc_208_check = supabase.table("lindex_documents").select(
                    "id, title, author, in_vector_store, collectionId"
                ).eq("id", 208).execute()
                
                if doc_208_check.data:
                    doc_208_data = doc_208_check.data[0]
                    collection_name = collection_map.get(str(doc_208_data.get("collectionId")), "Unknown")
                    logger.warning(f"Document 208 exists: title='{doc_208_data.get('title')}', author='{doc_208_data.get('author')}', collection='{collection_name}' (ID: {doc_208_data.get('collectionId')})")
                else:
                    logger.warning("Document 208 does not exist in lindex_documents table")
            
            metadata = {}
            for doc in filtered_docs:
                doc_id = str(doc["id"])
                collection_name = collection_map.get(str(doc.get("collectionId")), "Unknown")
                # Safely handle summary fields that might be None
                summary_short = doc.get("summary_short") or ""
                summary_medium = doc.get("summary_medium") or ""
                summary = summary_short or summary_medium
                if summary and len(summary) > 200:
                    summary = summary[:200]
                
                metadata[doc_id] = {
                    "summary": summary,
                    "title": doc.get("title", f"Document {doc_id}"),
                    "author": doc.get("author", "Unknown Author"),
                    "url": doc.get("url", ""),
                    "publish_date": doc.get("publish_date", ""),
                    "doc_size": doc.get("doc_size", 0),
                    "chunk_count": doc.get("chunk_count", 0),
                    "source_type": doc.get("source_type", "document"),
                    "collection_name": collection_name,  # Add collection name for debugging
                    "importance_weight": self._calculate_importance_weight(doc)
                }
                
                # Debug: Log specific documents we're looking for
                if doc_id in ["208", "212", "213"]:
                    logger.info(f"DEBUG - Document {doc_id}: title='{doc.get('title')}', author='{doc.get('author')}', collection='{collection_name}'")
                
            logger.info(f"Loaded metadata for {len(metadata)} documents")
            logger.info(f"Metadata keys: {list(metadata.keys())}")
            
            if len(metadata) == 0:
                logger.warning("⚠️ NO METADATA LOADED - This will cause 'Available metadata keys: []' error")
                logger.warning(f"   Collection name: {self.collection_name}")
                logger.warning(f"   Target collection ID: {target_collection_id}")
                logger.warning(f"   Filtered documents count: {len(filtered_docs)}")
                logger.warning(f"   Total vector store documents: {len(response.data)}")
                
                # Try to provide more diagnostic info
                if len(response.data) == 0:
                    logger.warning("   DIAGNOSIS: No documents found with in_vector_store = True")
                elif target_collection_id is None:
                    logger.warning(f"   DIAGNOSIS: Collection '{self.collection_name}' not found in available collections")
                else:
                    logger.warning(f"   DIAGNOSIS: No documents found in collection {target_collection_id}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading document metadata: {str(e)}")
            logger.error(f"Collection name: {self.collection_name}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    # The _calculate_importance_weight method stays the same as in your old code
    def _calculate_importance_weight(self, doc: Dict) -> float:
        """Calculate importance weight for documents (same as old code)."""
        doc_size = doc.get("doc_size", 0)
        chunk_count = doc.get("chunk_count", 0)
        source_type = doc.get("source_type", "document")
        
        size_weight = min(2.0, (doc_size / 50000))
        type_weights = {"book": 1.5, "pdf": 1.3, "document": 1.0, "email": 0.8}
        type_weight = type_weights.get(source_type, 1.0)
        chunk_weight = min(1.5, (chunk_count / 30))
        
        return size_weight * type_weight * chunk_weight


    def query_agentic_iterative(self, query: str, max_iterations: int = 3, llm_provider: str = DEFAULT_LLM_PROVIDER, balance_emphasis: str = None) -> Dict[str, Any]:
        """
        Iterative agentic query with multiple search rounds.
        An agent evaluates answer sufficiency and decides whether to continue searching.
        """
        start_time = time.time()
        logger.info(f"🔄 Iterative agentic query: {query}")
        try:
            # Get LLM
            llm_adapter = LLMProviderFactory.get_adapter(llm_provider)
            llm_instance = llm_adapter.get_llm_instance()
            agent_llm = llm_adapter.get_llm_instance()
            
            # Initialize tracking
            all_results = []
            iteration_history = []
            total_chunks_used = 0
            all_documents_used = {}
            
            for iteration in range(max_iterations):
                logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
                
                # Prepare iteration query
                if iteration == 0:
                    current_query = query
                else:
                    # Generate refined query based on previous results
                    refinement_prompt = f"""Based on the original question and the information found so far, 
                    generate a refined search query to find additional relevant information.
                    
                    Original question: {query}
                    
                    Information found so far:
                    {self._summarize_results(all_results)}
                    
                    Generate a short, focused search query for finding additional information:"""
                    
                    refined_query_response = llm_instance.complete(refinement_prompt)
                    current_query = str(refined_query_response).strip()
                    
                    if not current_query or len(current_query) < 5:
                        current_query = query  # Fallback to original
                
                logger.info(f"  Current query: {current_query}")
                
                # Perform search for this iteration
                retriever = self.vector_index.as_retriever(similarity_top_k=10)
                nodes = retriever.retrieve(current_query)
                
                if not nodes:
                    logger.info(f"  No results for iteration {iteration + 1}")
                    continue
                
                # Filter out chunks we've already used
                new_nodes = []
                for node in nodes:
                    chunk_id = node.metadata.get('chunk_index', -1)
                    doc_id = node.metadata.get('docid', 'unknown')
                    unique_id = f"{doc_id}_{chunk_id}"
                    
                    if unique_id not in [r.get('unique_id') for r in all_results]:
                        new_nodes.append(node)
                
                if not new_nodes:
                    logger.info(f"  No new chunks found in iteration {iteration + 1}")
                    continue
                
                # Take top new results
                selected_nodes = new_nodes[:5]
                
                # Track results
                for node in selected_nodes:
                    doc_id = node.metadata.get('docid', 'unknown')
                    chunk_id = node.metadata.get('chunk_index', -1)
                    
                    all_results.append({
                        'text': node.text,
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'unique_id': f"{doc_id}_{chunk_id}",
                        'iteration': iteration + 1,
                        'query_used': current_query,
                        'score': node.score
                    })
                    
                    # Track document usage
                    if doc_id not in all_documents_used:
                        all_documents_used[doc_id] = {
                            'chunks': 0,
                            'iterations_appeared': []
                        }
                    all_documents_used[doc_id]['chunks'] += 1
                    if iteration + 1 not in all_documents_used[doc_id]['iterations_appeared']:
                        all_documents_used[doc_id]['iterations_appeared'].append(iteration + 1)
                
                total_chunks_used += len(selected_nodes)
                
                # Record iteration
                iteration_history.append({
                    'iteration': iteration + 1,
                    'query': current_query,
                    'chunks_found': len(selected_nodes),
                    'new_chunks': len(new_nodes),
                    'cumulative_chunks': total_chunks_used
                })
                
                # Check if we have enough information
                if iteration < max_iterations - 1:  # Don't check on last iteration
                    sufficiency_prompt = f"""Evaluate if we have sufficient information to answer this question well:
                    "{query}"
                    
                    Information collected so far:
                    {self._summarize_results(all_results)}
                    
                    Do we have enough information for a comprehensive answer? Reply with only YES or NO."""
                    
                    sufficiency_response = llm_instance.complete(sufficiency_prompt)
                    is_sufficient = "yes" in str(sufficiency_response).lower()
                    
                    if is_sufficient:
                        logger.info(f"  ✅ Sufficient information found after {iteration + 1} iterations")
                        break
            
            # Generate final response
            if all_results:
                # Combine all information
                combined_context = "\n\n---\n\n".join([
                    f"[Iteration {r['iteration']}, Doc {r['doc_id']}]: {r['text']}"
                    for r in all_results
                ])
                
                final_prompt = f"""Based on all the information gathered through iterative search, 
                provide a comprehensive answer to: "{query}"
                
                Information gathered:
                {combined_context}
                
                Provide a detailed, well-structured answer that synthesizes all the relevant information found.
                
                ANSWER:"""
                
                response = llm_instance.complete(final_prompt)
                final_answer = str(response)
            else:
                final_answer = "I couldn't find relevant information to answer your question through iterative search."
            
            # Create document tracking
            documents_used = []
            source_attribution = []
            
            for doc_id, usage_info in all_documents_used.items():
                doc_metadata = self.document_metadata.get(str(doc_id), {})
                doc_title = doc_metadata.get('title', f'Document {doc_id}')
                doc_author = doc_metadata.get('author', 'Unknown Author')
                doc_url = doc_metadata.get('url', '')
                doc_publish_date = doc_metadata.get('publish_date', '')
                
                documents_used.append({
                    'document_id': doc_id,
                    'title': doc_title,
                    'author': doc_author,
                    'url': doc_url,
                    'publish_date': doc_publish_date,
                    'chunks_contributed': usage_info['chunks'],
                    'iterations_appeared': usage_info['iterations_appeared'],
                    'status': 'used'
                })
                
                source_attribution.append(
                    f"{doc_title} by {doc_author}: {usage_info['chunks']} chunks across iterations {usage_info['iterations_appeared']}"
                )
            
            documents_used.sort(key=lambda x: x['chunks_contributed'], reverse=True)
            
            total_time = time.time() - start_time
            
            # ✅ FIXED: Explicitly set searchQuality to premium for iterative method
            return {
                "status": "success",
                "query": query,
                "response": final_answer,
                "method": "agentic_iterative",
                "searchQuality": "premium",  # ✅ FIXED: Always premium for iterative
                "time_seconds": round(total_time, 2),
                "chunks_used": total_chunks_used,
                "documents_searched": len(all_documents_used),
                "documents_used": documents_used,
                "source_attribution": source_attribution,
                "iteration_history": iteration_history,
                "iterations_completed": len(iteration_history),
                "llm_provider": llm_provider,
                "embedding_dimension": embedding_manager.get_dimension()
            }

        except Exception as e:
            logger.error(f"❌ Iterative query error: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "searchQuality": "basic"  # ✅ FIXED: Set for error case too
            }

    def query_truly_agentic(self, query: str, llm_provider: str = DEFAULT_LLM_PROVIDER, balance_emphasis: str = None) -> Dict[str, Any]:
        """
        Truly agentic query that autonomously selects the best search strategy.
        Uses an agent to analyze the query and choose between simple, hybrid, or iterative approaches.
        """
        start_time = time.time()
        
        # Auto-detect balance emphasis if not provided
        if balance_emphasis is None:
            balance_emphasis = self._detect_query_type(query)
        
        logger.info(f"🤖🧠 Truly agentic query: {query} (balance: {balance_emphasis})")
        
        try:
            # Get LLM
            llm_adapter = LLMProviderFactory.get_adapter(llm_provider)
            llm_instance = llm_adapter.get_llm_instance()
            agent_llm = llm_adapter.get_llm_instance()
            
            # Step 1: Analyze query to determine best strategy
            strategy_prompt = f"""Analyze this query and determine the best search strategy:
            "{query}"
            
            Available strategies:
            1. SIMPLE: Best for straightforward factual questions
            2. HYBRID: Best for questions requiring comprehensive information from authoritative sources
            3. ITERATIVE: Best for complex questions that may need multiple search refinements
            
            Consider:
            - Query complexity
            - Need for comprehensive vs. quick answers
            - Whether the question has multiple aspects
            
            Respond with only one word: SIMPLE, HYBRID, or ITERATIVE."""
            
            strategy_response = llm_instance.complete(strategy_prompt)
            strategy = str(strategy_response).strip().upper()
            
            # Validate strategy
            valid_strategies = ['SIMPLE', 'HYBRID', 'ITERATIVE']
            if strategy not in valid_strategies:
                # Fallback based on query characteristics
                if len(query.split()) < 10 and '?' in query:
                    strategy = 'SIMPLE'
                elif any(term in query.lower() for term in ['comprehensive', 'detailed', 'all', 'everything']):
                    strategy = 'HYBRID'
                else:
                    strategy = 'ITERATIVE'
            
            logger.info(f"🎯 Selected strategy: {strategy}")
            
            # Step 2: Determine parameters based on query
            params = self._determine_search_parameters(query, strategy, llm_instance)
            logger.info(f"📊 Search parameters: {params}")
            
            # Step 3: Execute selected strategy
            if strategy == 'SIMPLE':
                result = self.query_simple(
                    query=query,
                    num_results=params.get('num_results', 5),
                    llm_provider=llm_provider,
                    balance_emphasis=balance_emphasis
                )
            elif strategy == 'HYBRID':
                result = self.query_hybrid_enhanced(
                    query=query,
                    top_k=params.get('top_k', 10),
                    llm_provider=llm_provider,
                    balance_emphasis=balance_emphasis
                )
            else:  # ITERATIVE
                result = self.query_agentic_iterative(
                    query=query,
                    max_iterations=params.get('max_iterations', 3),
                    llm_provider=llm_provider,
                    balance_emphasis=balance_emphasis
                )
            
            # Enhance result with meta-agent information
            if result.get('status') == 'success':
                result['method'] = f"truly_agentic_{strategy.lower()}"
                
                # ADD DYNAMIC searchQuality BASED ON SELECTED STRATEGY
                if strategy == 'SIMPLE':
                    result['searchQuality'] = "basic"
                elif strategy == 'HYBRID':
                    result['searchQuality'] = "advanced"  
                else:  # ITERATIVE
                    result['searchQuality'] = "premium"
                
                result['agent_reasoning'] = {
                    'selected_strategy': strategy,
                    'parameters_used': params,
                    'query_analysis': {
                        'length': len(query.split()),
                        'has_question_mark': '?' in query,
                        'complexity_keywords': sum(1 for term in ['how', 'why', 'what', 'when', 'where'] if term in query.lower())
                    }
                }
                result['time_seconds'] = round(time.time() - start_time, 2)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Truly agentic query error: {str(e)}")
            
            # Fallback to simple search
            logger.info("🔄 Falling back to simple search")
            try:
                fallback_result = self.query_simple(query, llm_provider=llm_provider)
                fallback_result['method'] = 'truly_agentic_fallback'
                fallback_result['searchQuality'] = "basic"
                fallback_result['fallback_reason'] = str(e)
                return fallback_result
            except Exception as fallback_error:
                return {"status": "error", "error": f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}"}



    def _summarize_results(self, results: List[Dict]) -> str:
        """Summarize results for iteration decisions."""
        if not results:
            return "No information found yet."
        
        summary_parts = []
        for i, result in enumerate(results[-5:]):  # Last 5 results
            text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            summary_parts.append(f"- From doc {result['doc_id']}: {text_preview}")
        
        return "\n".join(summary_parts)


    def _determine_search_parameters(self, query: str, strategy: str, llm_instance) -> Dict[str, Any]:
        """Determine optimal search parameters based on query analysis."""
        params = {}
        
        if strategy == 'SIMPLE':
            # Determine number of results needed
            if any(term in query.lower() for term in ['list', 'examples', 'multiple', 'various']):
                params['num_results'] = 10
            else:
                params['num_results'] = 5
                
        elif strategy == 'HYBRID':
            # Determine top_k based on query breadth
            if any(term in query.lower() for term in ['comprehensive', 'complete', 'all', 'everything']):
                params['top_k'] = 15
            else:
                params['top_k'] = 10
                
        else:  # ITERATIVE
            # Determine iterations based on complexity
            complexity_indicators = ['how', 'why', 'explain', 'analyze', 'compare']
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in query.lower())
            
            if complexity_score >= 2:
                params['max_iterations'] = 4
            else:
                params['max_iterations'] = 3
        
        return params

    def _calculate_importance_weight(self, doc: Dict, balance_emphasis: str = "comprehensive") -> float:
        """Calculate importance weight for documents with configurable balance."""
        doc_size = doc.get("doc_size", 0)
        chunk_count = doc.get("chunk_count", 0)
        source_type = doc.get("source_type", "document")
        
        # Base weights for different document types
        type_weights = {"book": 1.5, "pdf": 1.3, "document": 1.0, "email": 0.8, "article": 0.9, "news": 0.85}
        base_type_weight = type_weights.get(source_type, 1.0)
        
        # Apply balance emphasis
        if balance_emphasis == "news_focused":
            # Favor shorter, more recent content
            type_weights = {"book": 0.8, "pdf": 0.9, "document": 1.0, "email": 0.7, "article": 1.2, "news": 1.3}
            base_type_weight = type_weights.get(source_type, 1.0)
            size_weight = min(1.2, (doc_size / 10000))  # Less aggressive size weighting
            chunk_weight = min(1.1, (chunk_count / 20))  # Less chunk bias
        elif balance_emphasis == "balanced":
            # Moderate balance between comprehensive and recent
            type_weights = {"book": 1.2, "pdf": 1.1, "document": 1.0, "email": 0.8, "article": 1.1, "news": 1.1}
            base_type_weight = type_weights.get(source_type, 1.0)
            size_weight = min(1.5, (doc_size / 30000))  # Moderate size weighting
            chunk_weight = min(1.3, (chunk_count / 25))  # Moderate chunk bias
        else:  # "comprehensive" (default)
            # Original behavior - favor comprehensive sources
            size_weight = min(2.0, (doc_size / 50000))
            chunk_weight = min(1.5, (chunk_count / 30))
        
        return size_weight * base_type_weight * chunk_weight

    def _detect_query_type(self, query: str) -> str:
        """Detect query type to automatically determine balance emphasis."""
        query_lower = query.lower()
        
        # News and current events indicators
        news_indicators = [
            "latest", "recent", "current", "today", "this week", "this month",
            "2024", "2025", "breaking", "news", "update", "trending",
            "happening now", "just released", "announced", "reported"
        ]
        
        # Comprehensive research indicators
        comprehensive_indicators = [
            "comprehensive", "detailed", "complete", "thorough", "in-depth",
            "everything about", "all about", "full analysis", "complete guide",
            "comprehensive study", "extensive", "exhaustive"
        ]
        
        news_score = sum(1 for indicator in news_indicators if indicator in query_lower)
        comprehensive_score = sum(1 for indicator in comprehensive_indicators if indicator in query_lower)
        
        if news_score >= 2 or any(indicator in query_lower for indicator in ["latest", "recent", "current", "2024", "2025"]):
            return "news_focused"
        elif comprehensive_score >= 2 or any(indicator in query_lower for indicator in ["comprehensive", "detailed", "everything"]):
            return "comprehensive"
        else:
            return "balanced"

    def query_simple(self, query: str, num_results: int = 5, llm_provider: str = DEFAULT_LLM_PROVIDER, balance_emphasis: str = None) -> Dict[str, Any]:
        """Simple similarity search with proper document tracking."""
        start_time = time.time()
        
        # Auto-detect balance emphasis if not provided
        if balance_emphasis is None:
            balance_emphasis = self._detect_query_type(query)
        
        logger.info(f"🔍 Simple query: {query} (balance: {balance_emphasis})")
        
        try:
            # Get LLM
            llm_adapter = LLMProviderFactory.get_adapter(llm_provider)
            llm_instance = llm_adapter.get_llm_instance()
            
            # Perform retrieval - simplified without ThreadPoolExecutor for simple queries
            retriever = self.vector_index.as_retriever(similarity_top_k=num_results)
            
            logger.info(f"  Retrieving {num_results} chunks...")
            nodes = retriever.retrieve(query)
            logger.info(f"  Retrieved {len(nodes)} chunks")
            
            if not nodes:
                return {
                    "status": "success",
                    "query": query,
                    "response": "No relevant information found for your query.",
                    "method": "simple_similarity",
                    "searchQuality": "basic",  # ✅ FIXED: Set for empty results
                    "time_seconds": round(time.time() - start_time, 2),
                    "chunks_used": 0,
                    "documents_searched": 0,
                    "documents_used": [],
                    "source_attribution": [],
                    "llm_provider": llm_provider,
                    "embedding_dimension": embedding_manager.get_dimension()
                }
            
            # Apply importance weighting with balance emphasis
            weighted_nodes = []
            doc_scores = {}
            
            for node in nodes:
                doc_id = node.metadata.get('docid', '')
                if doc_id:
                    # Get document metadata and calculate importance weight
                    doc_metadata = self.document_metadata.get(str(doc_id), {})
                    weight = self._calculate_importance_weight(doc_metadata, balance_emphasis)
                    
                    # Apply weight to similarity score
                    boosted_score = node.score * weight
                    
                    # Track best score per document
                    if doc_id not in doc_scores or boosted_score > doc_scores[doc_id]['score']:
                        doc_scores[doc_id] = {
                            'score': boosted_score,
                            'weight': weight,
                            'original_score': node.score
                        }
                    
                    # Update node score
                    node.score = boosted_score
                    weighted_nodes.append(node)
            
            # Sort by boosted scores and take top results
            weighted_nodes.sort(key=lambda x: x.score, reverse=True)
            selected_nodes = weighted_nodes[:num_results]
            
            # Track document usage
            document_usage = {}
            context_parts = []
            
            for i, node in enumerate(selected_nodes):
                doc_id = node.metadata.get('docid', 'unknown')
                
                # Track usage
                if doc_id != 'unknown':
                    if doc_id not in document_usage:
                        document_usage[doc_id] = {
                            'chunks': 0,
                            'total_score': 0.0,
                            'characters': 0,
                            'weight': doc_scores.get(doc_id, {}).get('weight', 1.0)
                        }
                    document_usage[doc_id]['chunks'] += 1
                    document_usage[doc_id]['total_score'] += node.score
                    document_usage[doc_id]['characters'] += len(node.text)
                
                # Add to context
                context_parts.append(node.text)
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"  Created context from {len(context_parts)} chunks, {len(context)} chars")
            
            # Generate response - simplified prompt
            prompt = f"""Answer the following question based on the provided context.
            
    QUESTION: {query}

    CONTEXT:
    {context}

    ANSWER:"""
            
            logger.info(f"  Generating LLM response...")
            response = llm_instance.complete(prompt)
            logger.info(f"  LLM response generated")
            
            # Create document tracking
            documents_used = []
            source_attribution = []
            
            for doc_id, usage_info in document_usage.items():
                doc_metadata = self.document_metadata.get(str(doc_id), {})
                
                # If metadata not found or has default values, try direct lookup
                if not doc_metadata or doc_metadata.get('title') == f'Document {doc_id}':
                    logger.warning(f"QUERY_SIMPLE - Metadata not found for document {doc_id}, attempting direct lookup...")
                    logger.warning(f"QUERY_SIMPLE - Current collection_name: {self.collection_name}")
                    logger.warning(f"QUERY_SIMPLE - Available metadata keys: {list(self.document_metadata.keys())}")
                    try:
                        # Use HTTPSupabaseClient for direct lookup to avoid compatibility issues
                        from knowledge_gap_http_supabase import HTTPSupabaseClient
                        http_supabase = HTTPSupabaseClient()
                        direct_response = http_supabase.table("lindex_documents").select(
                            "id, title, author, url, publish_date, in_vector_store, collectionId"
                        ).eq("id", doc_id).execute()
                        
                        if direct_response.data:
                            direct_doc = direct_response.data[0]
                            doc_title = direct_doc.get('title', f'Document {doc_id}')
                            doc_author = direct_doc.get('author', 'Unknown Author')
                            doc_url = direct_doc.get('url', '')
                            doc_publish_date = direct_doc.get('publish_date', '')
                            collection_id = direct_doc.get('collectionId')
                            logger.info(f"QUERY_SIMPLE - Direct lookup SUCCESS for doc {doc_id}: title='{doc_title}', author='{doc_author}', url='{doc_url}', publish_date='{doc_publish_date}', collectionId={collection_id}")
                        else:
                            doc_title = f'Document {doc_id}'
                            doc_author = 'Unknown Author'
                            doc_url = ''
                            doc_publish_date = ''
                            logger.warning(f"QUERY_SIMPLE - Document {doc_id} not found in database")
                    except Exception as e:
                        logger.error(f"QUERY_SIMPLE - Error in direct lookup for document {doc_id}: {e}")
                        doc_title = f'Document {doc_id}'
                        doc_author = 'Unknown Author'
                        doc_url = ''
                        doc_publish_date = ''
                else:
                    doc_title = doc_metadata.get('title', f'Document {doc_id}')
                    doc_author = doc_metadata.get('author', 'Unknown Author')
                    doc_url = doc_metadata.get('url', '')
                    doc_publish_date = doc_metadata.get('publish_date', '')
                    logger.info(f"QUERY_SIMPLE - Using cached metadata for doc {doc_id}: title='{doc_title}', author='{doc_author}', url='{doc_url}', publish_date='{doc_publish_date}'")
                
                # Debug: Log what we're getting for each document
                logger.info(f"QUERY_SIMPLE - QUERY DEBUG - Document {doc_id}: metadata={doc_metadata}, title='{doc_title}', author='{doc_author}'")
                
                documents_used.append({
                    'document_id': doc_id,
                    'title': doc_title,
                    'author': doc_author,
                    'url': doc_url,
                    'publish_date': doc_publish_date,
                    'chunks_contributed': usage_info['chunks'],
                    'characters_retrieved': usage_info['characters'],
                    'average_score': round(usage_info['total_score'] / usage_info['chunks'], 4),
                    'status': 'used'
                })
                
                source_attribution.append(f"{doc_title} by {doc_author}: {usage_info['chunks']} chunks")
            
            documents_used.sort(key=lambda x: x['chunks_contributed'], reverse=True)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Simple query completed in {total_time:.2f}s")
            
            return {
                "status": "success",
                "query": query,
                "response": str(response),
                "method": "simple_similarity",
                "searchQuality": "basic",  # ✅ FIXED: Explicitly set to basic
                "time_seconds": round(total_time, 2),
                "chunks_used": len(nodes),
                "documents_searched": len(document_usage),
                "documents_used": documents_used,
                "source_attribution": source_attribution,
                "llm_provider": llm_provider,
                "embedding_dimension": embedding_manager.get_dimension()
            }
            
        except Exception as e:
            logger.error(f"❌ Simple query error: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "searchQuality": "basic"  # ✅ FIXED: Set even for errors
            }
    def query_agentic_fixed(self, query: str, max_docs: int = 3, llm_provider: str = DEFAULT_LLM_PROVIDER, verbose_mode: str = "balanced", balance_emphasis: str = None) -> Dict[str, Any]:
        """ENHANCED agentic query with MULTI-TOOL ENFORCEMENT - Prevents single-document responses."""
        start_time = time.time()
        logger.info(f"🤖 ENHANCED MULTI-TOOL AGENTIC query: {query}")
        logger.info(f"   Using LLM provider: {llm_provider}, verbose mode: {verbose_mode}")
        
        if not AGENTS_AVAILABLE:
            logger.warning("⚠️ Agents not available, falling back to simple search")
            fallback_result = self.query_simple(query, num_results=10, llm_provider=llm_provider)
            fallback_result['searchQuality'] = "basic"
            return fallback_result
        
        try:
            # Get available documents first
            available_docs = list(self.document_metadata.keys())[:max_docs + 2]
            
            if not available_docs:
                return {
                    "status": "success",
                    "query": query,
                    "response": "No documents available in the collection.",
                    "method": "agentic_fixed",
                    "searchQuality": "basic",
                    "time_seconds": round(time.time() - start_time, 2),
                    "llm_provider": llm_provider,
                    "embedding_dimension": embedding_manager.get_dimension()
                }
            
            # Get LLM instance for agent
            llm_adapter = LLMProviderFactory.get_adapter(llm_provider)
            
            # ENHANCED: Better LLM settings for multi-tool usage
            if verbose_mode == "concise":
                agent_llm = llm_adapter.get_llm_instance()
                if llm_provider == "deepseek":
                    agent_llm.temperature = 0.3
                    agent_llm.max_tokens = 1000
                elif llm_provider == "gemini":
                    agent_llm.temperature = 0.3
                    agent_llm.max_tokens = 1000
                similarity_top_k = 8
                response_mode = "tree_summarize"
                max_iterations = 12
                min_length = 150
                
            elif verbose_mode == "detailed":
                agent_llm = llm_adapter.get_llm_instance()
                if llm_provider == "deepseek":
                    agent_llm.temperature = 0.4
                    agent_llm.max_tokens = 2000
                elif llm_provider == "gemini":
                    agent_llm.temperature = 0.4
                    agent_llm.max_tokens = 2000
                similarity_top_k = 10
                response_mode = "tree_summarize"
                max_iterations = 15
                min_length = 300
                
            else:  # "balanced" - default
                agent_llm = llm_adapter.get_llm_instance()
                if llm_provider == "deepseek":
                    agent_llm.temperature = 0.4
                    agent_llm.max_tokens = 1500
                elif llm_provider == "gemini":
                    agent_llm.temperature = 0.4
                    agent_llm.max_tokens = 1500
                similarity_top_k = 8
                response_mode = "tree_summarize"
                max_iterations = 12
                min_length = 250
            
            # Create document tools with ENHANCED DESCRIPTIONS
            document_tools = []
            working_tools = 0
            
            for doc_id in available_docs[:max_docs]:
                doc_info = self.document_metadata.get(str(doc_id), {})
                title = doc_info.get('title', f'Document {doc_id}')
                summary = doc_info.get('summary', '')
                
                logger.info(f"🔧 Creating enhanced tool for '{title}' (ID: {doc_id})")
                
                try:
                    # Create proper filters
                    filters = MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="docid",
                                value=str(doc_id),
                                operator=FilterOperator.EQ
                            )
                        ]
                    )
                    
                    # Create query engine with error handling
                    doc_query_engine = self.vector_index.as_query_engine(
                        similarity_top_k=similarity_top_k,
                        filters=filters,
                        llm=agent_llm,
                        response_mode=response_mode
                    )
                    
                    # Test tool reliability
                    try:
                        test_response = doc_query_engine.query("test content")
                        test_str = str(test_response).strip()
                        
                        if not test_str or len(test_str) < 10:
                            logger.warning(f"  ⚠️ Tool for '{title}' returns short response, trying without filters")
                            doc_query_engine = self.vector_index.as_query_engine(
                                similarity_top_k=similarity_top_k,
                                llm=agent_llm,
                                response_mode=response_mode
                            )
                            test_response = doc_query_engine.query("test content")
                            test_str = str(test_response).strip()
                            
                            if not test_str or len(test_str) < 10:
                                logger.warning(f"  ❌ Tool for '{title}' still not working, skipping")
                                continue
                                
                    except Exception as test_error:
                        logger.warning(f"  ❌ Tool test failed for '{title}': {test_error}")
                        continue
                    
                    # ENHANCED: Better tool descriptions that encourage usage
                    tool_name = f"search_{doc_id}"
                    
                    # Smart description based on document content
                    if any(keyword in title.lower() for keyword in ['rental', 'investing', 'investment', 'property']):
                        tool_description = f"🏢 INVESTMENT EXPERT: '{title}' - Essential source for real estate investment strategies, rental property analysis, financing options, market insights, and ROI calculations. Critical for comprehensive investment guidance and strategy questions."
                    elif any(keyword in title.lower() for keyword in ['home', 'buying', 'buyer', 'purchase']):
                        tool_description = f"🏠 HOME BUYING SPECIALIST: '{title}' - Complete guide covering residential real estate, market conditions, financing options, legal considerations, and buyer strategies. Essential for home purchase and market analysis questions."
                    elif any(keyword in title.lower() for keyword in ['landscape', 'garden', 'property improvement']):
                        tool_description = f"🌿 PROPERTY VALUE EXPERT: '{title}' - Property enhancement through landscaping, curb appeal optimization, and outdoor space development. Valuable for property value improvement and aesthetic enhancement strategies."
                    else:
                        tool_description = f"📚 REAL ESTATE RESOURCE: '{title}' - Comprehensive real estate information covering {summary[:120]}. Important source for thorough real estate knowledge and specialized topics."
                    
                    tool = QueryEngineTool.from_defaults(
                        query_engine=doc_query_engine,
                        name=tool_name,
                        description=tool_description
                    )
                    
                    document_tools.append(tool)
                    working_tools += 1
                    logger.info(f"  ✅ Created enhanced tool for '{title}' with strategic description")
                    
                except Exception as e:
                    logger.warning(f"  ❌ Failed to create tool for '{title}': {e}")
                    continue
            
            if working_tools == 0:
                logger.error("❌ No working document tools created - falling back to simple search")
                fallback_result = self.query_simple(query, num_results=8, llm_provider=llm_provider)
                fallback_result['searchQuality'] = "basic"
                return fallback_result
            
            logger.info(f"✅ Created {working_tools} enhanced document tools")
            
            # FORCE multi-tool research instead of using agent
            logger.info(f"🔥 FORCING multi-tool research with {len(document_tools)} tools")
            
            def execute_forced_multi_tool_research():
                """Force search of multiple tools and synthesize results - with smart tool selection"""
                all_results = []
                tools_used = []
                
                # SMART TOOL SELECTION: Determine relevance based on query and tool descriptions
                query_lower = query.lower()
                relevant_tools = []
                
                for tool in document_tools:
                    tool_desc = tool.metadata.description.lower()
                    tool_name = tool.metadata.name.lower()
                    
                    # Check if tool is relevant to the query
                    relevance_score = 0
                    
                    # High relevance keywords
                    if any(keyword in query_lower for keyword in ['invest', 'strategy', 'strategies', 'property', 'rental', 'real estate']):
                        if any(keyword in tool_desc for keyword in ['invest', 'rental', 'property', 'strategy']):
                            relevance_score += 3
                        if any(keyword in tool_desc for keyword in ['home', 'buying', 'market']):
                            relevance_score += 2
                        if any(keyword in tool_desc for keyword in ['landscape', 'garden']):
                            relevance_score += 1 if 'property' in tool_desc else 0
                    
                    # Always include if it's an investment/rental focused tool
                    if any(keyword in tool_desc for keyword in ['investment', 'rental property', 'investing']):
                        relevance_score += 3
                        
                    relevant_tools.append((tool, relevance_score))
                    logger.info(f"  📊 Tool {tool.metadata.name}: relevance score {relevance_score}")
                
                # Sort by relevance and take top tools (minimum 1, maximum 3)
                relevant_tools.sort(key=lambda x: x[1], reverse=True)
                selected_tools = [tool for tool, score in relevant_tools if score > 0][:3]
                
                # If no relevant tools, use all tools as fallback
                if not selected_tools:
                    selected_tools = document_tools
                    logger.warning("⚠️ No clearly relevant tools found, using all tools")
                else:
                    logger.info(f"🎯 Selected {len(selected_tools)} relevant tools based on query analysis")
                
                # Force search each relevant tool with different query variations
                for i, tool in enumerate(selected_tools):
                    tool_name = tool.metadata.name
                    logger.info(f"🔍 SMART search {i+1}/{len(selected_tools)}: {tool_name}")
                    
                    try:
                        # Vary the query for each tool to get different perspectives
                        if i == 0:
                            search_query = f"{query}"
                        elif i == 1:
                            search_query = f"{query} different approaches methods strategies"
                        else:
                            search_query = f"{query} additional considerations options"
                        
                        result = tool.call(search_query)
                        result_text = str(result).strip()
                        
                        # More lenient threshold for accepting results
                        if result_text and len(result_text) > 50:
                            all_results.append({
                                'tool': tool_name,
                                'content': result_text,
                                'query': search_query
                            })
                            tools_used.append(tool_name)
                            logger.info(f"  ✅ Got {len(result_text)} chars from {tool_name}")
                        else:
                            logger.warning(f"  ⚠️ Poor result from {tool_name} ({len(result_text)} chars)")
                            
                    except Exception as e:
                        logger.warning(f"  ❌ Error with {tool_name}: {e}")
                        continue
                
                if not all_results:
                    raise Exception("No tools returned valid results")
                
                # Enhanced synthesis with better handling of single vs multiple sources
                if len(all_results) == 1:
                    # Single source - enhance it with more comprehensive analysis
                    single_result = all_results[0]
                    enhancement_prompt = f"""Based on this comprehensive real estate resource, provide a detailed analysis of the question:

    QUESTION: {query}

    SOURCE CONTENT: {single_result['content']}

    ENHANCEMENT REQUIREMENTS:
    - Expand into a comprehensive {min_length}-400 word response
    - Break down the strategies into clear categories
    - Add practical examples and implementation details
    - Include pros/cons or considerations for each strategy
    - Provide actionable advice for beginners and experienced investors
    - Organize information in a logical, easy-to-follow structure

    COMPREHENSIVE DETAILED RESPONSE ({min_length}-400 words):"""

                    try:
                        enhanced = agent_llm.complete(enhancement_prompt)
                        enhanced_text = str(enhanced).strip()
                        
                        logger.info(f"🎯 Enhanced single source into {len(enhanced_text)} chars")
                        
                        return {
                            'response': enhanced_text,
                            'sources_used': tools_used,
                            'synthesis_method': 'single_source_enhanced',
                            'tools_searched': len(all_results),
                            'raw_results': all_results
                        }
                        
                    except Exception as e:
                        logger.error(f"Single source enhancement failed: {e}")
                        return {
                            'response': single_result['content'],
                            'sources_used': tools_used,
                            'synthesis_method': 'single_source_raw',
                            'tools_searched': len(all_results),
                            'raw_results': all_results
                        }
                
                else:
                    # Multiple sources - synthesize them
                    synthesis_prompt = f"""You are synthesizing information from multiple real estate expert sources to answer this question comprehensively:

    QUESTION: {query}

    SOURCES AND INFORMATION:
    {chr(10).join([f"SOURCE {i+1} - {result['tool']}:{chr(10)}{result['content']}{chr(10)}" for i, result in enumerate(all_results)])}

    SYNTHESIS REQUIREMENTS:
    - Create a comprehensive {min_length}-400 word response
    - Integrate information from ALL {len(all_results)} sources above
    - Organize into clear categories or types of strategies
    - Include specific examples and practical details from the sources
    - Demonstrate synthesis across multiple expert perspectives
    - Provide actionable, evidence-based advice

    COMPREHENSIVE SYNTHESIZED RESPONSE ({min_length}-400 words):"""

                    try:
                        synthesized = agent_llm.complete(synthesis_prompt)
                        synthesized_text = str(synthesized).strip()
                        
                        logger.info(f"🎯 Synthesized {len(synthesized_text)} chars from {len(all_results)} sources")
                        
                        return {
                            'response': synthesized_text,
                            'sources_used': tools_used,
                            'synthesis_method': 'multi_source_synthesized',
                            'tools_searched': len(all_results),
                            'raw_results': all_results
                        }
                        
                    except Exception as e:
                        logger.error(f"Multi-source synthesis failed: {e}")
                        # Fallback: combine raw results
                        combined = f"Based on comprehensive research across {len(all_results)} expert sources:\n\n"
                        for i, result in enumerate(all_results):
                            combined += f"From {result['tool']}:\n{result['content']}\n\n"
                        
                        return {
                            'response': combined,
                            'sources_used': tools_used,
                            'synthesis_method': 'multi_source_combined',
                            'tools_searched': len(all_results),
                            'raw_results': all_results
                        }
            
            # Execute forced multi-tool research
            try:
                multi_tool_result = execute_forced_multi_tool_research()
                
                response_text = multi_tool_result['response']
                tools_used_names = multi_tool_result['sources_used']
                
                # Determine search quality based on synthesis method
                if multi_tool_result['synthesis_method'] == 'single_source_enhanced':
                    search_quality = "advanced"  # Enhanced single source
                elif multi_tool_result['synthesis_method'] == 'multi_source_synthesized':
                    search_quality = "premium"   # Multi-source synthesis
                else:
                    search_quality = "advanced"  # Multi-tool research
                
                logger.info(f"✅ FORCED multi-tool completed:")
                logger.info(f"   📊 Response length: {len(response_text)} chars")
                logger.info(f"   🔧 Tools used: {', '.join(tools_used_names)}")
                logger.info(f"   📚 Sources: {multi_tool_result['tools_searched']} tools searched")
                
                # Create fake agent_response object for compatibility with rest of code
                class FakeAgentResponse:
                    def __init__(self, text, sources):
                        self.response = text
                        self.source_nodes = []
                        # Create fake source nodes for each tool used
                        for i, tool_name in enumerate(sources):
                            doc_id = tool_name.split('_')[-1]  # Extract doc_id from tool name
                            fake_node = type('obj', (object,), {
                                'metadata': {'docid': doc_id},
                                'text': f"Content from {tool_name}"
                            })
                            self.source_nodes.append(fake_node)
                    
                    def __str__(self):
                        return self.response
                
                agent_response = FakeAgentResponse(response_text, tools_used_names)
                
                # Validate response length and enhance if needed
                if len(response_text) < min_length:
                    logger.warning(f"⚠️ Still too short ({len(response_text)} chars), final enhancement needed")
                    
                    final_enhance_prompt = f"""This response needs to be expanded to meet the {min_length}-400 word requirement:

    {response_text}

    Expand this into a comprehensive {min_length}-400 word response with:
    - More detailed explanations of each strategy
    - Specific examples and practical applications  
    - Additional context and considerations
    - Actionable advice and implementation steps

    EXPANDED COMPREHENSIVE RESPONSE ({min_length}-400 words):"""
                    
                    try:
                        final_enhanced = agent_llm.complete(final_enhance_prompt)
                        final_text = str(final_enhanced).strip()
                        
                        if final_text and len(final_text) > len(response_text):
                            response_text = final_text
                            agent_response = FakeAgentResponse(response_text, tools_used_names)
                            logger.info(f"✅ Final enhancement: {len(response_text)} chars")
                    except Exception as e:
                        logger.warning(f"Final enhancement failed: {e}")
                
            except Exception as multi_tool_error:
                logger.error(f"❌ Forced multi-tool failed: {multi_tool_error}")
                # Fall back to simple search
                fallback_result = self.query_simple(query, num_results=8, llm_provider=llm_provider)
                fallback_result['searchQuality'] = "basic"
                fallback_result['method'] = f"agentic_fixed_{verbose_mode}_forced_fallback"
                fallback_result['fallback_reason'] = f"Forced multi-tool error: {str(multi_tool_error)}"
                return fallback_result
            
            # ENHANCED: Response validation and enhancement
            response_text = str(agent_response).strip()
            
            # Validate response meets multi-source requirements
            if len(response_text) < min_length:
                logger.warning(f"⚠️ Response too short ({len(response_text)} chars), enhancing with multi-source synthesis...")
                
                # Multi-source enhancement prompt
                enhance_prompt = f"""The previous response was too brief and may not demonstrate comprehensive multi-source research.

    ORIGINAL QUESTION: {query}

    BRIEF RESPONSE: {response_text}

    ENHANCEMENT REQUIREMENTS:
    - Create a comprehensive {min_length}-400 word response
    - Incorporate information from multiple real estate sources/perspectives
    - Provide specific strategies, examples, and practical details
    - Demonstrate thorough research across different approaches
    - Include actionable, evidence-based advice
    - Show synthesis of multiple information sources

    COMPREHENSIVE ENHANCED RESPONSE ({min_length}-400 words):"""
                
                try:
                    enhanced_response = agent_llm.complete(enhance_prompt)
                    enhanced_str = str(enhanced_response).strip()
                    
                    if enhanced_str and len(enhanced_str) > len(response_text) and len(enhanced_str) >= min_length:
                        response_text = enhanced_str
                        logger.info(f"✅ Response enhanced from {len(str(agent_response))} to {len(response_text)} chars")
                    else:
                        logger.warning("⚠️ Enhancement insufficient, using fallback")
                        fallback_result = self.query_simple(query, num_results=8, llm_provider=llm_provider)
                        fallback_result['searchQuality'] = "basic"
                        fallback_result['method'] = f"agentic_fixed_{verbose_mode}_enhancement_fallback"
                        fallback_result['fallback_reason'] = f"Response too short ({len(response_text)} chars) and enhancement insufficient"
                        return fallback_result
                        
                except Exception as e:
                    logger.warning(f"Enhancement failed: {e}, using fallback")
                    fallback_result = self.query_simple(query, num_results=8, llm_provider=llm_provider)
                    fallback_result['searchQuality'] = "basic"
                    fallback_result['method'] = f"agentic_fixed_{verbose_mode}_enhancement_error"
                    fallback_result['fallback_reason'] = f"Response too short and enhancement error: {str(e)}"
                    return fallback_result
            
            # Extract document usage and source tracking
            chunks_used = len(getattr(agent_response, 'source_nodes', []))
            
            documents_used = []
            source_attribution = []
            
            if hasattr(agent_response, 'source_nodes') and agent_response.source_nodes:
                doc_usage = {}
                for node in agent_response.source_nodes:
                    doc_id = str(node.metadata.get('docid', 'unknown'))
                    if doc_id in self.document_metadata:
                        if doc_id not in doc_usage:
                            doc_usage[doc_id] = {'chunks': 0, 'characters': 0}
                        doc_usage[doc_id]['chunks'] += 1
                        doc_usage[doc_id]['characters'] += len(node.text)
                
                for doc_id, usage in doc_usage.items():
                    doc_info = self.document_metadata[doc_id]
                    documents_used.append({
                        'document_id': doc_id,
                        'title': doc_info['title'],
                        'author': doc_info.get('author', 'Unknown Author'),
                        'url': doc_info.get('url', ''),
                        'publish_date': doc_info.get('publish_date', ''),
                        'chunks_contributed': usage['chunks'],
                        'characters_retrieved': usage['characters'],
                        'status': 'used'
                    })
                    source_attribution.append(f"{doc_info['title']} by {doc_info.get('author', 'Unknown Author')}: {usage['chunks']} chunks")
            
            total_time = time.time() - start_time
            
            # Final validation
            if not response_text or len(response_text.strip()) < 50:
                logger.error("❌ Final response validation failed - using emergency fallback")
                emergency_result = self.query_simple(query, num_results=8, llm_provider=llm_provider)
                emergency_result['searchQuality'] = "basic"
                emergency_result['method'] = f"agentic_fixed_{verbose_mode}_emergency_fallback"
                emergency_result['fallback_reason'] = "Final response validation failed"
                return emergency_result
            
            # SUCCESS: Log comprehensive results
            logger.info(f"✅ ENHANCED multi-tool agentic query completed successfully:")
            logger.info(f"   📊 Response length: {len(response_text)} chars (target: {min_length}+)")
            logger.info(f"   📚 Tools available: {working_tools}, Chunks used: {chunks_used}")
            logger.info(f"   ⏱️ Time: {total_time:.2f}s, Mode: {verbose_mode}")
            logger.info(f"   🎯 Multi-source research enforced with {max_iterations} max iterations")
            
            if source_attribution:
                logger.info(f"   🔗 Sources: {'; '.join(source_attribution)}")
            
            return {
                "status": "success",
                "query": query,
                "response": response_text,
                "method": f"agentic_fixed_{verbose_mode}_enhanced",
                "searchQuality": search_quality,  # Now properly set based on synthesis method
                "time_seconds": round(total_time, 2),
                "chunks_used": chunks_used,
                "documents_searched": len(document_tools),
                "documents_used": documents_used,
                "source_attribution": source_attribution,
                "agent_info": {
                    'agent_type': 'ReActAgent_MultiTool_Enhanced',
                    'verbosity_mode': verbose_mode,
                    'tools_available': len(document_tools),
                    'working_tools': working_tools,
                    'max_iterations': max_iterations,
                    'similarity_top_k': similarity_top_k,
                    'response_mode': response_mode,
                    'final_response_length': len(response_text),
                    'minimum_length_enforced': min_length,
                    'multi_tool_enforcement': True,
                    'max_tokens': getattr(agent_llm, 'max_tokens', 'unknown')
                },
                "llm_provider": llm_provider,
                "embedding_dimension": embedding_manager.get_dimension()
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced agentic query error: {str(e)}")
            
            # Emergency fallback on any error
            logger.info("🔄 Using emergency fallback to simple search")
            try:
                emergency_result = self.query_simple(query, num_results=8, llm_provider=llm_provider)
                emergency_result['method'] = f"agentic_fixed_{verbose_mode}_error_fallback"
                emergency_result['searchQuality'] = "basic"
                emergency_result['fallback_reason'] = str(e)
                return emergency_result
            except Exception as fallback_error:
                return {"status": "error", "error": f"Primary error: {str(e)}, Fallback error: {str(fallback_error)}"}


def get_rag_engine(collection_name: str) -> SimpleRAGEngine:
    """Get or create RAG engine for collection."""
    global _engines
    
    if collection_name not in _engines:
        logger.info(f"🔧 Creating new RAG engine for collection: {collection_name}")
        _engines[collection_name] = SimpleRAGEngine(collection_name)
    else:
        logger.info(f"📁 Using cached RAG engine for collection: {collection_name}")
        # Force refresh metadata to ensure we have the latest data
        logger.info(f"🔄 Refreshing metadata for collection: {collection_name}")
        try:
            _engines[collection_name].document_metadata = _engines[collection_name]._load_document_metadata()
            logger.info(f"✅ Metadata refresh completed. Loaded {len(_engines[collection_name].document_metadata)} documents")
        except Exception as e:
            logger.error(f"❌ Error refreshing metadata: {e}")
    
    return _engines[collection_name]


def _process_document_background(docid, collection_name, source_type="document", extra_metadata=None):
    """Background task for processing documents."""
    try:
        supabase.table("lindex_documents").update({
            "processing_status": "processing"
        }).eq("id", docid).execute()
        
        embedding_manager.ensure_optimizer_fitted()
        
        doc_response = supabase.table("lindex_documents").select(
            "parsedText"
        ).eq("id", docid).execute()
        
        if not doc_response.data or not doc_response.data[0].get("parsedText"):
            logger.error(f"Document {docid} has no parsable content")
            return
            
        document_text = doc_response.data[0]["parsedText"]
        
        result = document_processor.process_document(
            docid=docid, 
            collection_name=collection_name,
            source_type=source_type,
            extra_metadata=extra_metadata
        )
        
        if result.get('success', False):
            doc_size = len(document_text)
            chunk_count = result.get('embedding', {}).get('total_nodes', 0)
            
            supabase.table("lindex_documents").update({
                "processing_status": "completed",
                "in_vector_store": True,
                "last_processed": datetime.now().isoformat(),
                "source_type": source_type,
                "doc_size": doc_size,
                "chunk_count": chunk_count
            }).eq("id", docid).execute()
        else:
            error_msg = result.get('error', 'Unknown error')
            supabase.table("lindex_documents").update({
                "processing_status": "error",
                "error_message": error_msg[:200]
            }).eq("id", docid).execute()
            
    except Exception as e:
        logger.exception(f"Background processing error for document {docid}: {str(e)}")
        
        try:
            supabase.table("lindex_documents").update({
                "processing_status": "error",
                "error_message": str(e)[:200]
            }).eq("id", docid).execute()
        except:
            pass
## ******************>
def _execute_async_query(job_id, data):
    """Execute query in background with status updates supporting all methods."""
    start_time = time.time()
    
    try:
        job = active_jobs[job_id]
        
        # Initialize
        job.update_status("running", 5, "Initializing query engine...", 
                         phase="initialization",
                         llm_provider=data.get('llm', DEFAULT_LLM_PROVIDER),
                         search_strategy=data.get('method', 'simple'))
        
        query = data.get('query', '')
        collection_name = data.get('collection_name', '')
        method = data.get('method', 'simple')
        llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
        
        # Load engine
        job.update_status("running", 15, "Loading RAG engine...", 
                         phase="engine_setup",
                         embedding_dimension=embedding_manager.get_dimension())
        
        engine = get_rag_engine(collection_name)
        
        # Document analysis
        job.update_status("running", 25, "Analyzing document collection...", 
                         phase="document_analysis")
        
        available_docs = list(engine.document_metadata.keys())
        doc_info = []
        if available_docs:
            for doc_id in available_docs[:5]:
                doc_meta = engine.document_metadata.get(doc_id, {})
                title = doc_meta.get('title', f'Document {doc_id}')
                doc_info.append(f"{title} (ID: {doc_id})")
        
        job.update_status("running", 30, f"Found {len(available_docs)} documents", 
                         documents_being_searched=doc_info)
        
        # Execute query based on method
        job.update_status("running", 40, f"Starting {method} search...", 
                         phase="search_execution",
                         search_strategy=method)
        
        result = None
        
        # ✅ CRITICAL FIX: Ensure each method call preserves searchQuality
        if method == 'simple':
            num_results = data.get('num_results', 5)
            job.update_status("running", 45, f"Simple search with {num_results} results...")
            result = engine.query_simple(query, num_results=num_results, llm_provider=llm_provider)
            
        elif method == 'hybrid_enhanced':
            top_k = data.get('top_k', 10)
            job.update_status("running", 45, f"Hybrid search with importance weighting (top {top_k})...")
            result = engine.query_hybrid_enhanced(query, top_k=top_k, llm_provider=llm_provider)
            
        elif method == 'agentic_fixed':
            max_docs = data.get('max_docs', 3)
            verbose_mode = data.get('verbose_mode', 'balanced')
            job.update_status("running", 45, f"Creating agent with {max_docs} document tools...")
            result = engine.query_agentic_fixed(query, max_docs=max_docs, llm_provider=llm_provider, verbose_mode=verbose_mode)
            
        elif method == 'agentic_iterative':
            max_iterations = data.get('max_iterations', 3)
            job.update_status("running", 45, f"Iterative search (max {max_iterations} rounds)...")
            result = engine.query_agentic_iterative(query, max_iterations=max_iterations, llm_provider=llm_provider)
            
        elif method == 'truly_agentic':
            job.update_status("running", 45, "Agent analyzing query and selecting strategy...")
            result = engine.query_truly_agentic(query, llm_provider=llm_provider)
        else:
            # Default to simple
            result = engine.query_simple(query, llm_provider=llm_provider)
        
        # ✅ CRITICAL DEBUG: Log the raw result before processing
        if result:
            logger.info(f"🔍 RAW RESULT from {method}:")
            logger.info(f"   Status: {result.get('status', 'NO_STATUS')}")
            logger.info(f"   SearchQuality: {result.get('searchQuality', 'NOT_FOUND')}")
            logger.info(f"   Method: {result.get('method', 'NO_METHOD')}")
            logger.info(f"   Keys: {list(result.keys())}")
        else:
            logger.error(f"❌ {method} returned None result!")
        
        # Process results
        job.update_status("running", 80, "Processing search results...", 
                         phase="result_processing")
        
        if result and isinstance(result, dict) and result.get('status') == 'success':
            chunks_used = result.get('chunks_used', 0)
            docs_searched = result.get('documents_searched', 0)
            
            # 🔧 CRITICAL FIX: Extract searchQuality EXACTLY as returned by method
            search_quality = result.get('searchQuality')
            
            # ✅ VALIDATION: Ensure we have a valid searchQuality
            if search_quality is None:
                logger.error(f"❌ Method {method} returned None searchQuality! Setting to basic")
                search_quality = 'basic'
            elif search_quality not in ['basic', 'advanced', 'premium']:
                logger.warning(f"⚠️ Method {method} returned invalid searchQuality: {search_quality}. Setting to basic")
                search_quality = 'basic'
            
            # Log the final search quality for debugging
            logger.info(f"🎯 FINAL searchQuality for {method}: {search_quality}")
            
            job.update_status("running", 95, "Generating final response...", 
                             phase="response_generation",
                             chunks_processed=chunks_used,
                             results_quality=search_quality)
            
            # 🔧 CRITICAL FIX: Store the EXACT searchQuality from the method
            job.result = {
                'response': result.get('response', ''),
                'method': result.get('method', method),
                'searchQuality': search_quality,  # ✅ Use validated searchQuality
                'time_seconds': result.get('time_seconds', 0),
                'chunks_used': chunks_used,
                'documents_searched': docs_searched,
                'embedding_dimension': result.get('embedding_dimension', embedding_manager.get_dimension()),
                'documents_used': result.get('documents_used', []),
                'source_attribution': result.get('source_attribution', []),
                'llm_provider': llm_provider
            }
            
            # Add method-specific data
            if method == 'hybrid_enhanced' and result.get('enhancement_info'):
                job.result['enhancement_info'] = result['enhancement_info']
            elif method == 'agentic_iterative' and result.get('iteration_history'):
                job.result['iteration_history'] = result['iteration_history']
                job.result['iterations_completed'] = result.get('iterations_completed', 0)
            elif method == 'truly_agentic' and result.get('agent_reasoning'):
                job.result['agent_reasoning'] = result['agent_reasoning']
            elif method == 'agentic_fixed' and result.get('agent_info'):
                job.result['agent_info'] = result['agent_info']
            
            job.update_status("completed", 100, 
                             f"Query completed! Used {chunks_used} chunks from {docs_searched} documents.",
                             phase="completed",
                             results_quality=search_quality)
            
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            job.error = error_msg
            job.update_status("failed", 0, f"Query failed: {error_msg}", phase="failed")
            
    except Exception as e:
        logger.exception(f"Async query execution error for job {job_id}: {str(e)}")
        job = active_jobs.get(job_id)
        if job:
            job.error = str(e)
            job.update_status("failed", 0, f"Execution error: {str(e)}", phase="failed")
    
    # Clean up after 10 minutes
    def cleanup():
        time.sleep(600)
        if job_id in active_jobs:
            del active_jobs[job_id]
            logger.info(f"Cleaned up job {job_id}")
    
    Thread(target=cleanup, daemon=True).start()
    
##**************************************

def create_app():
    """Create Flask application with error handling and memory management."""
    
    # Add memory monitoring
    import psutil
    import gc
    
    def log_memory_usage():
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"🧠 Memory usage: {memory_mb:.1f} MB")
        return memory_mb
    
    try:
        logger.info("🚀 Starting Flask application creation...")
        log_memory_usage()
        app = Flask(__name__)
        app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
        app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        logger.info("🔧 Creating Flask application...")
        
        # ✅ FIXED: Make sure document_processor is available at module level
        import __main__
        __main__.document_processor = document_processor
        __main__.background_executor = background_executor
   
        @app.route('/health', methods=['GET'])
        def health_check():
            """Simple health check endpoint for Docker/load balancers"""
            return jsonify({
                "status": "healthy",
                "service": "rag-system",
                "timestamp": datetime.now().isoformat()
            }), 200
        
        @app.route('/embedding_status', methods=['GET'])
        def embedding_status():
            """Get current embedding configuration status."""
            try:
                model = embedding_manager.get_model()
                dimension = embedding_manager.get_dimension()
                
                status = {
                    "optimization_enabled": os.getenv("USE_EMBEDDING_OPTIMIZATION", "false").lower() == "true",
                    "optimization_available": OPTIMIZATION_AVAILABLE,
                    "current_dimension": dimension,
                    "model_type": "optimized" if hasattr(model, 'optimizer') else "standard",
                    "optimizer_fitted": getattr(model, 'optimizer', None) and model.optimizer.fitted if hasattr(model, 'optimizer') else True,
                    "agents_available": AGENTS_AVAILABLE
                }
                
                return jsonify(status)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        

        # Add this to your main.py to test Linkup directly:
        @app.route('/test_linkup', methods=['GET'])
        def test_linkup_connection():
            """Test Linkup API connection and response format."""
            try:
                # Test parameters
                test_query = request.args.get('query', 'real estate market trends')
                
                # Import and test
                from knowledge_gap_http_supabase import MultiSourceResearcher, HTTPSupabaseClient
                
                supabase_client = HTTPSupabaseClient()
                researcher = MultiSourceResearcher(supabase_client)
                
                if not researcher.linkup_client:
                    return jsonify({
                        "status": "error",
                        "error": "Linkup client not initialized",
                        "linkup_available": False,
                        "api_key_set": bool(os.getenv('LINKUP_API_KEY'))
                    })
                
                # Test the connection
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(
                    researcher._linkup_search(test_query, max_results=3)
                )
                
                return jsonify({
                    "status": "success",
                    "test_query": test_query,
                    "linkup_available": True,
                    "api_key_set": bool(os.getenv('LINKUP_API_KEY')),
                    "results_found": len(results),
                    "results": results,
                    "linkup_client_type": str(type(researcher.linkup_client))
                })
                
            except Exception as e:
                import traceback
                return jsonify({
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "linkup_available": False
                }), 500


        
        @app.route('/upload', methods=['POST'])
        def upload_file():
            """Upload and parse a document file."""
            try:
                start_time = time.time()
                
                if 'file' not in request.files or 'docid' not in request.form:
                    return jsonify({"error": "Missing file or docid"}), 400

                file = request.files['file']
                docid = request.form['docid']
                collection_name = request.form.get('collection_name', 'default_collection')
                
                # Parse metadata if provided
                extra_metadata = {}
                metadata_str = request.form.get('metadata')
                if metadata_str:
                    try:
                        extra_metadata = json.loads(metadata_str)
                        logger.info(f"Received metadata for doc {docid}: {extra_metadata.keys()}")
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata for doc {docid}: {e}")

                logger.info(f"Upload started for document {docid}, collection = {collection_name}")

                if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
                    return jsonify({"error": "Unsupported file type"}), 400

                file_extension = file.filename.split('.')[-1].lower()
                source_type = file_extension
                
                filename = secure_filename(file.filename)
                local_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(local_file_path)
                
                supabase.table("lindex_documents").update({
                    "processing_status": "uploading",
                    "source_type": source_type
                }).eq("id", docid).execute()
                
                try:
                    documents = SimpleDirectoryReader(
                        input_files=[local_file_path],
                        file_extractor={
                            ".pdf": parser,
                            ".docx": parser,
                            ".txt": None
                        }
                    ).load_data()
                    
                    full_parsed_text = ""
                    for doc in documents:
                        full_parsed_text += doc.text + "\n\n"
                        
                    text_length = len(full_parsed_text)
                    
                    if text_length < 100:
                        raise ValueError(f"Text too short ({text_length} chars). Parsing failed.")
                        
                    supabase.table("lindex_documents").update({
                        "parsedText": full_parsed_text,
                        "processing_status": "parsed",
                        "source_type": source_type,
                        "last_processed": datetime.now().isoformat()
                    }).eq("id", docid).execute()
                    
                    os.remove(local_file_path)
                    
                    background_executor.submit(
                        _process_document_background,
                        docid,
                        collection_name,
                        source_type,
                        extra_metadata
                    )
                    
                    processing_time = time.time() - start_time
                    
                    return jsonify({
                        "message": f"File parsed successfully in {processing_time:.2f}s",
                        "docid": docid,
                        "text_length": text_length,
                        "processing_status": "parsed",
                        "source_type": source_type,
                        "background_processing": "started",
                        "embedding_dimension": embedding_manager.get_dimension()
                    }), 200
                    
                except Exception as e:
                    logger.exception(f"Parsing error for document {docid}: {str(e)}")
                    
                    supabase.table("lindex_documents").update({
                        "processing_status": "error",
                        "error_message": str(e)[:200]
                    }).eq("id", docid).execute()
                    
                    if os.path.exists(local_file_path):
                        os.remove(local_file_path)
                        
                    return jsonify({
                        "error": f"Parsing failed: {str(e)}",
                        "docid": docid
                    }), 500
                    
            except Exception as e:
                logger.exception(f"Upload error: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/chunk', methods=['POST'])
        def chunk_document():
            """Process document chunks and store in vector database."""
            try:
                data = request.get_json()
                if not data or 'docid' not in data:
                    return jsonify({"error": "Missing docid in request"}), 400

                docid = data['docid']
                collection_name = data.get('collection_name', 'default_collection')
                source_type = data.get('source_type', 'document')
                
                logger.info(f"Chunking document {docid} for collection {collection_name}")
                
                background_executor.submit(
                    _process_document_background,
                    docid,
                    collection_name,
                    source_type
                )
                
                return jsonify({
                    "message": "Document processing started in background",
                    "docid": docid,
                    "collection_name": collection_name,
                    "source_type": source_type,
                    "embedding_dimension": embedding_manager.get_dimension()
                }), 202
                
            except Exception as e:
                logger.exception(f"Chunking error: {str(e)}")
                return jsonify({"error": str(e)}), 500
  
        @app.route('/generate_summary', methods=['POST'])
        def handle_generate_summary():
            """Generate document summary from chunks."""
            try:
                data = request.get_json()
                if not data or 'docid' not in data or 'collection_name' not in data:
                    return jsonify({
                        "status": "error",
                        "error": "Missing required fields (docid, collection_name)"
                    }), 400
                    
                docid = data['docid']
                collection_name = data['collection_name']
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                
                logger.info(f"📝 Starting summary generation for document {docid}")
                
                # Submit to background executor
                background_executor.submit(
                    _generate_summary_background,
                    docid,
                    collection_name,
                    llm_provider
                )
                
                return jsonify({
                    "status": "success",
                    "message": "Summary generation started in background",
                    "docid": docid,
                    "collection_name": collection_name,
                    "llm_provider": llm_provider,
                    "embedding_dimension": embedding_manager.get_dimension()
                }), 202
                
            except Exception as e:
                logger.exception(f"Summary generation error: {str(e)}")
                return jsonify({
                    "status": "error",
                    "error": str(e)
                }), 500 
        @app.route('/query_hybrid_enhanced', methods=['POST'])
        def query_hybrid_enhanced_endpoint():
            """Enhanced hybrid query endpoint with importance weighting."""
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                collection_name = data.get('collection_name', '').strip()
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                top_k = data.get('top_k', 10)
                balance_emphasis = data.get('balance_emphasis', None)  # New parameter
                
                if not query:
                    return jsonify({"status": "error", "error": "Query is required"}), 400
                if not collection_name:
                    return jsonify({"status": "error", "error": "Collection name is required"}), 400
                
                engine = get_rag_engine(collection_name)
                result = engine.query_hybrid_enhanced(query, top_k=top_k, llm_provider=llm_provider, balance_emphasis=balance_emphasis)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in hybrid enhanced query endpoint: {str(e)}")
                return jsonify({"status": "error", "error": str(e)}), 500


        @app.route('/query_agentic_iterative', methods=['POST'])
        def query_agentic_iterative_endpoint():
            """Iterative agentic query endpoint with multiple search rounds."""
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                collection_name = data.get('collection_name', '').strip()
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                max_iterations = data.get('max_iterations', 3)
                
                if not query:
                    return jsonify({"status": "error", "error": "Query is required"}), 400
                if not collection_name:
                    return jsonify({"status": "error", "error": "Collection name is required"}), 400
                
                engine = get_rag_engine(collection_name)
                result = engine.query_agentic_iterative(query, max_iterations=max_iterations, llm_provider=llm_provider)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in agentic iterative query endpoint: {str(e)}")
                return jsonify({"status": "error", "error": str(e)}), 500


        @app.route('/query_truly_agentic', methods=['POST'])
        def query_truly_agentic_endpoint():
            """Truly agentic query endpoint that autonomously selects the best strategy."""
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                collection_name = data.get('collection_name', '').strip()
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                
                if not query:
                    return jsonify({"status": "error", "error": "Query is required"}), 400
                if not collection_name:
                    return jsonify({"status": "error", "error": "Collection name is required"}), 400
                
                engine = get_rag_engine(collection_name)
                result = engine.query_truly_agentic(query, llm_provider=llm_provider)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in truly agentic query endpoint: {str(e)}")
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/query_simple', methods=['POST'])
        def query_simple_endpoint():
            """Simple query endpoint using basic similarity search."""
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                collection_name = data.get('collection_name', '').strip()
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                num_results = data.get('num_results', 5)
                balance_emphasis = data.get('balance_emphasis', None)  # New parameter
                
                if not query:
                    return jsonify({"status": "error", "error": "Query is required"}), 400
                if not collection_name:
                    return jsonify({"status": "error", "error": "Collection name is required"}), 400
                
                engine = get_rag_engine(collection_name)
                result = engine.query_simple(query, num_results=num_results, llm_provider=llm_provider, balance_emphasis=balance_emphasis)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in simple query endpoint: {str(e)}")
                return jsonify({"status": "error", "error": str(e)}), 500
        
        @app.route('/query_agentic_fixed', methods=['POST'])
        def query_agentic_fixed_endpoint():
            """Fixed agentic query endpoint with verbosity control."""
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                collection_name = data.get('collection_name', '').strip()
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                max_docs = data.get('max_docs', 3)
                verbose_mode = data.get('verbose_mode', 'balanced')  # NEW: verbosity control
                
                # Validate verbose_mode
                if verbose_mode not in ['concise', 'balanced', 'detailed']:
                    verbose_mode = 'balanced'
                
                if not query:
                    return jsonify({"status": "error", "error": "Query is required"}), 400
                if not collection_name:
                    return jsonify({"status": "error", "error": "Collection name is required"}), 400
                
                engine = get_rag_engine(collection_name)
                result = engine.query_agentic_fixed(
                    query, 
                    max_docs=max_docs, 
                    llm_provider=llm_provider,
                    verbose_mode=verbose_mode  # NEW: pass verbosity mode
                )
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in agentic fixed query endpoint: {str(e)}")
                return jsonify({"status": "error", "error": str(e)}), 500
        @app.route('/process_gap_filler_docs_manual', methods=['POST'])
        def process_gap_filler_docs_manual():
            """Manual processing for gap filler documents with compatibility fixes."""
            try:
                data = request.get_json() or {}
                user_id = data.get('user_id')
                collection_name = data.get('collection_name', 'default_collection')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                logger.info(f"🔧 Manual processing of gap filler documents for user: {user_id}")
                
                # Get gap filler documents that need processing
                docs_response = supabase.table("lindex_documents").select(
                    "id, title, processing_status, source_type, user_id"
                ).eq("source_type", "gap_filler").eq("user_id", user_id).in_("processing_status", ["pending", "pending_processor"]).execute()
                
                if not docs_response.data:
                    return jsonify({
                        "status": "success",
                        "message": f"No pending or pending_processor gap filler documents found for user {user_id}",
                        "processed_count": 0
                    })
                
                pending_docs = docs_response.data
                processed_count = 0
                error_count = 0
                results = []
                
                for doc in pending_docs:
                    doc_id = doc.get('id')
                    doc_title = doc.get('title', 'Unknown')
                    
                    try:
                        logger.info(f"  Processing gap filler doc {doc_id}: {doc_title[:50]}...")
                        
                        # Update status
                        supabase.table("lindex_documents").update({
                            "processing_status": "processing"
                        }).eq("id", doc_id).execute()
                        
                        # Try processing with enhanced error handling
                        try:
                            result = document_processor.process_document(
                                docid=doc_id,
                                collection_name=collection_name,
                                source_type="gap_filler"
                            )
                            
                            if result.get('success', False):
                                # Success - update to completed
                                chunk_count = result.get('embedding', {}).get('total_nodes', 0)
                                
                                supabase.table("lindex_documents").update({
                                    "processing_status": "completed",
                                    "in_vector_store": True,
                                    "chunk_count": chunk_count,
                                    "last_processed": datetime.now().isoformat()
                                }).eq("id", doc_id).execute()
                                
                                processed_count += 1
                                results.append({
                                    "doc_id": doc_id,
                                    "title": doc_title,
                                    "status": "success",
                                    "chunks": chunk_count
                                })
                                
                                logger.info(f"    ✅ Successfully processed {doc_id} with {chunk_count} chunks")
                                
                            else:
                                error_msg = result.get('error', 'Processing failed')
                                raise Exception(error_msg)
                                
                        except Exception as proc_error:
                            error_msg = str(proc_error)
                            
                            # Handle specific compatibility issues
                            if "SyncPostgrestClient" in error_msg or "http_client" in error_msg:
                                supabase.table("lindex_documents").update({
                                    "processing_status": "pending_processor",
                                    "error_message": "Supabase compatibility issue"
                                }).eq("id", doc_id).execute()
                                
                                results.append({
                                    "doc_id": doc_id,
                                    "title": doc_title,
                                    "status": "compatibility_issue",
                                    "error": "Supabase client compatibility"
                                })
                                
                            else:
                                supabase.table("lindex_documents").update({
                                    "processing_status": "error",
                                    "error_message": error_msg[:200]
                                }).eq("id", doc_id).execute()
                                
                                error_count += 1
                                results.append({
                                    "doc_id": doc_id,
                                    "title": doc_title,
                                    "status": "failed",
                                    "error": error_msg
                                })
                                
                    except Exception as e:
                        logger.error(f"    ❌ Error processing {doc_id}: {e}")
                        error_count += 1
                        
                        try:
                            supabase.table("lindex_documents").update({
                                "processing_status": "error",
                                "error_message": str(e)[:200]
                            }).eq("id", doc_id).execute()
                        except:
                            pass
                        
                        results.append({
                            "doc_id": doc_id,
                            "title": doc_title,
                            "status": "error",
                            "error": str(e)
                        })
                
                return jsonify({
                    "status": "success",
                    "message": f"Manual processing completed: {processed_count} successful, {error_count} failed",
                    "total_docs": len(pending_docs),
                    "processed_count": processed_count,
                    "error_count": error_count,
                    "results": results,
                    "user_id": user_id
                })
                
            except Exception as e:
                logger.error(f"Error in manual gap filler processing: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500
            
        @app.route('/query_async', methods=['POST'])
        def query_async():
            """Start an async query job."""
            try:
                data = request.json
                job_id = str(uuid.uuid4())
                collection_name = data.get('collection_name')
                query = data.get('query')
                method = data.get('method', 'simple')
                llm_provider = data.get('llm', DEFAULT_LLM_PROVIDER)
                balance_emphasis = data.get('balance_emphasis', None)  # New parameter
                
                logger.info(f"Starting async job {job_id} for query: {query}...")
                logger.info(f"Method: {method}, LLM: {llm_provider}")
                
                # Store job info
                job_info = {
                    'status': 'processing',
                    'phase': 'initialization',
                    'progress': 0,
                    'current_step': 'Starting query...',
                    'search_info': {
                        'strategy': method,
                        'llm_provider': llm_provider,
                        'embedding_dimension': 1536,
                        'documents_being_searched': [],
                        'current_document': '',
                        'chunks_processed': 0,
                        'search_iterations': 0,
                        'max_iterations': 0,
                        'results_quality': 'unknown'
                    },
                    'steps': []
                }
                jobs[job_id] = job_info
                
                # Run query in background thread
                def run_query():
                    try:
                        # Update status
                        job_info['phase'] = 'engine_setup'
                        job_info['progress'] = 10
                        job_info['current_step'] = 'Setting up RAG engine...'
                        
                        # Get or create RAG engine
                        rag_engine = get_rag_engine(collection_name)
                        
                        # Update status
                        job_info['phase'] = 'search_execution'
                        job_info['progress'] = 20
                        job_info['current_step'] = f'Executing {method} search...'
                        
                        # Execute query based on method with balance emphasis
                        if method == 'simple':
                            result = rag_engine.query_simple(
                                query, 
                                num_results=data.get('num_results', 5),
                                llm_provider=llm_provider,
                                balance_emphasis=balance_emphasis
                            )
                        elif method == 'hybrid_enhanced':
                            result = rag_engine.query_hybrid_enhanced(
                                query,
                                top_k=data.get('top_k', 10),
                                llm_provider=llm_provider,
                                balance_emphasis=balance_emphasis
                            )
                        elif method == 'agentic_fixed':
                            result = rag_engine.query_agentic_fixed(
                                query,
                                max_docs=data.get('max_docs', 3),
                                llm_provider=llm_provider,
                                verbose_mode=data.get('verbose_mode', 'balanced'),
                                balance_emphasis=balance_emphasis
                            )
                        elif method == 'agentic_iterative':
                            result = rag_engine.query_agentic_iterative(
                                query,
                                max_iterations=data.get('max_iterations', 3),
                                llm_provider=llm_provider,
                                balance_emphasis=balance_emphasis
                            )
                        elif method == 'truly_agentic':
                            result = rag_engine.query_truly_agentic(
                                query,
                                llm_provider=llm_provider,
                                balance_emphasis=balance_emphasis
                            )
                        else:
                            raise ValueError(f"Unknown method: {method}")
                        
                        # Update final status
                        job_info['status'] = 'completed'
                        job_info['phase'] = 'completed'
                        job_info['progress'] = 100
                        job_info['current_step'] = 'Query completed successfully'
                        job_info['result'] = result
                        
                    except Exception as e:
                        logger.error(f"Query job {job_id} failed: {str(e)}")
                        job_info['status'] = 'failed'
                        job_info['phase'] = 'failed'
                        job_info['error'] = str(e)
                        job_info['current_step'] = f'Error: {str(e)}'
                
                # Start background thread
                thread = threading.Thread(target=run_query)
                thread.start()
                
                return jsonify({'job_id': job_id})
                
            except Exception as e:
                logger.error(f"Failed to start async query: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @app.route('/gap_filler_documents/<title_id>', methods=['GET'])
        def get_gap_filler_documents_by_title(title_id):
            """Get all gap filler documents associated with a specific title."""
            try:
                user_id = request.args.get('user_id')
                
                logger.info(f"📋 Getting gap filler documents for title: {title_id}")
                
                from knowledge_gap_http_supabase import HTTPSupabaseClient
                
                http_supabase = HTTPSupabaseClient()
                
                # Get gap filler documents for this title
                docs_response = http_supabase.execute_query(
                    'GET',
                    f'lindex_documents?select=id,title,source_type,data_source_type,processing_status,in_vector_store,chunk_count,focus_keyword,research_timestamp,summary_short,url,importance_score&title_id=eq.{title_id}&source_type=eq.gap_filler'
                )
                
                if not docs_response['success']:
                    return jsonify({
                        "status": "error",
                        "error": "Failed to fetch gap filler documents",
                        "details": docs_response.get('error')
                    }), 500
                
                gap_docs = docs_response['data']
                
                # Get the title information
                title_response = http_supabase.execute_query(
                    'GET',
                    f'Titles?select=id,Title,focus_keyword,status&id=eq.{title_id}'
                )
                
                title_info = {}
                if title_response['success'] and title_response['data']:
                    title_info = title_response['data'][0]
                
                # Organize documents by data source type
                docs_by_type = {}
                total_docs = len(gap_docs)
                docs_in_vector = 0
                total_chunks = 0
                
                for doc in gap_docs:
                    data_source_type = doc.get('data_source_type', 'unknown')
                    
                    if data_source_type not in docs_by_type:
                        docs_by_type[data_source_type] = []
                    
                    docs_by_type[data_source_type].append({
                        'document_id': doc.get('id'),
                        'title': doc.get('title', 'Unknown'),
                        'processing_status': doc.get('processing_status', 'unknown'),
                        'in_vector_store': doc.get('in_vector_store', False),
                        'chunk_count': doc.get('chunk_count', 0),
                        'summary': doc.get('summary_short', ''),
                        'url': doc.get('url', ''),
                        'importance_score': doc.get('importance_score', 0),
                        'research_timestamp': doc.get('research_timestamp', '')
                    })
                    
                    if doc.get('in_vector_store', False):
                        docs_in_vector += 1
                    
                    total_chunks += doc.get('chunk_count', 0)
                
                # Calculate effectiveness
                vector_effectiveness = round((docs_in_vector / total_docs) * 100, 1) if total_docs > 0 else 0
                
                return jsonify({
                    "status": "success",
                    "title_id": title_id,
                    "title_info": title_info,
                    "user_id": user_id,
                    "summary": {
                        "total_gap_filler_docs": total_docs,
                        "documents_in_vector_store": docs_in_vector,
                        "total_chunks": total_chunks,
                        "vector_effectiveness": f"{vector_effectiveness}%",
                        "data_source_types": list(docs_by_type.keys()),
                        "source_type_counts": {k: len(v) for k, v in docs_by_type.items()}
                    },
                    "gap_filler_documents": docs_by_type,
                    "query_timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting gap filler documents for title {title_id}: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500

        # =============================================================================
        # New Endpoint: Get All Title IDs with Gap Filler Documents
        # =============================================================================

        @app.route('/titles_with_gap_fillers', methods=['GET'])
        def get_titles_with_gap_fillers():
            """Get all title IDs that have gap filler documents."""
            try:
                user_id = request.args.get('user_id')
                
                logger.info(f"📊 Getting all titles with gap filler documents for user: {user_id}")
                
                from knowledge_gap_http_supabase import HTTPSupabaseClient
                
                http_supabase = HTTPSupabaseClient()
                
                # Get distinct title IDs that have gap filler documents
                docs_response = http_supabase.execute_query(
                    'GET',
                    'lindex_documents?select=title_id,data_source_type,processing_status,in_vector_store&source_type=eq.gap_filler'
                )
                
                if not docs_response['success']:
                    return jsonify({
                        "status": "error",
                        "error": "Failed to fetch gap filler documents",
                        "details": docs_response.get('error')
                    }), 500
                
                gap_docs = docs_response['data']
                
                # Group by title_id
                titles_summary = {}
                
                for doc in gap_docs:
                    title_id = doc.get('title_id')
                    if not title_id:
                        continue
                    
                    if title_id not in titles_summary:
                        titles_summary[title_id] = {
                            'title_id': title_id,
                            'total_docs': 0,
                            'docs_in_vector': 0,
                            'data_source_types': set(),
                            'processing_status_counts': {}
                        }
                    
                    titles_summary[title_id]['total_docs'] += 1
                    
                    if doc.get('in_vector_store', False):
                        titles_summary[title_id]['docs_in_vector'] += 1
                    
                    data_source_type = doc.get('data_source_type', 'unknown')
                    titles_summary[title_id]['data_source_types'].add(data_source_type)
                    
                    processing_status = doc.get('processing_status', 'unknown')
                    status_counts = titles_summary[title_id]['processing_status_counts']
                    status_counts[processing_status] = status_counts.get(processing_status, 0) + 1
                
                # Convert sets to lists and calculate effectiveness
                for title_id, summary in titles_summary.items():
                    summary['data_source_types'] = list(summary['data_source_types'])
                    summary['vector_effectiveness'] = round(
                        (summary['docs_in_vector'] / summary['total_docs']) * 100, 1
                    ) if summary['total_docs'] > 0 else 0
                
                # Get title names
                if titles_summary:
                    title_ids = list(titles_summary.keys())
                    title_ids_str = ','.join(title_ids)
                    
                    titles_response = http_supabase.execute_query(
                        'GET',
                        f'Titles?select=id,Title,status&id=in.({title_ids_str})'
                    )
                    
                    if titles_response['success']:
                        for title_data in titles_response['data']:
                            title_id = title_data.get('id')
                            if title_id in titles_summary:
                                titles_summary[title_id]['title_name'] = title_data.get('Title', 'Unknown')
                                titles_summary[title_id]['title_status'] = title_data.get('status', 'Unknown')
                
                # Convert to list and sort by total docs
                titles_list = list(titles_summary.values())
                titles_list.sort(key=lambda x: x['total_docs'], reverse=True)
                
                return jsonify({
                    "status": "success",
                    "user_id": user_id,
                    "query_timestamp": datetime.now().isoformat(),
                    "summary": {
                        "titles_with_gap_fillers": len(titles_list),
                        "total_gap_filler_docs": sum(t['total_docs'] for t in titles_list),
                        "total_docs_in_vector": sum(t['docs_in_vector'] for t in titles_list)
                    },
                    "titles": titles_list
                })
                
            except Exception as e:
                logger.error(f"Error getting titles with gap fillers: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500


        @app.route('/check_gap_filler_vector_status', methods=['GET'])
        def check_gap_filler_vector_status():
            """Check vector store status specifically for gap filler documents."""
            try:
                user_id = request.args.get('user_id')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                # Get all gap filler documents
                docs_response = supabase.table("lindex_documents").select(
                    "id, title, processing_status, in_vector_store, source_type, error_message, chunk_count"
                ).eq("source_type", "gap_filler").execute()
                
                if not docs_response.data:
                    return jsonify({
                        "status": "success",
                        "message": "No gap filler documents found",
                        "total_gap_docs": 0
                    })
                
                gap_docs = docs_response.data
                
                # Categorize by status
                status_counts = {
                    "completed": 0,
                    "pending": 0,
                    "processing": 0,
                    "error": 0,
                    "pending_processor": 0,
                    "in_vector_store": 0
                }
                
                compatibility_issues = []
                
                for doc in gap_docs:
                    status = doc.get("processing_status", "unknown")
                    in_vector = doc.get("in_vector_store", False)
                    error_msg = doc.get("error_message", "")
                    
                    if status in status_counts:
                        status_counts[status] += 1
                    
                    if in_vector:
                        status_counts["in_vector_store"] += 1
                    
                    # Check for compatibility issues
                    if "SyncPostgrestClient" in error_msg or "http_client" in error_msg or "compatibility" in error_msg:
                        compatibility_issues.append({
                            "doc_id": doc.get("id"),
                            "title": doc.get("title", "Unknown")[:50],
                            "error": error_msg
                        })
                
                total_docs = len(gap_docs)
                vector_effectiveness = round((status_counts["in_vector_store"] / total_docs) * 100, 1) if total_docs > 0 else 0
                
                return jsonify({
                    "status": "success",
                    "user_id": user_id,
                    "query_timestamp": datetime.now().isoformat(),
                    "gap_filler_summary": {
                        "total_gap_filler_docs": total_docs,
                        "status_breakdown": status_counts,
                        "vector_store_effectiveness": f"{vector_effectiveness}%",
                        "compatibility_issues_count": len(compatibility_issues)
                    },
                    "compatibility_issues": compatibility_issues,
                    "recommendations": {
                        "needs_manual_processing": status_counts["pending"] > 0,
                        "has_compatibility_issues": len(compatibility_issues) > 0,
                        "vector_store_ready": status_counts["in_vector_store"] > 0,
                        "next_action": "run_manual_processing" if status_counts["pending"] > 0 else "check_compatibility"
                    }
                })
                
            except Exception as e:
                logger.error(f"Error checking gap filler vector status: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500



        @app.route('/enhance_knowledge/<title_id>', methods=['POST'])
        def enhance_single_title(title_id):
            """Enhance knowledge for a single title with user_id from request body."""
            try:
                data = request.get_json() or {}
                user_id = data.get('user_id')  # GET FROM REQUEST BODY
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                # Collection control parameters
                collection_name = data.get('collection_name')
                merge_with_existing = data.get('merge_with_existing', True)
                
                logger.info(f"🎯 Enhancing title {title_id} for user {user_id}")
                logger.info(f"   - Requested collection: {collection_name or 'auto-detect'}")
                logger.info(f"   - Merge with existing: {merge_with_existing}")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # First get the title data and analyze the gap
                response = orchestrator.supabase.table("titles").select("*").eq("id", title_id).execute()
                
                if not response.data:
                    return jsonify({"status": "error", "error": f"Title {title_id} not found"}), 404
                
                title_data = response.data[0]
                
                # Analyze knowledge gap
                gap = loop.run_until_complete(
                    orchestrator.gap_analyzer._analyze_single_title(title_data)
                )
                
                if not gap:
                    return jsonify({
                        'status': 'no_gap',
                        'message': 'No knowledge gap identified'
                    })
                
                # Research gap
                research_results = loop.run_until_complete(
                    orchestrator.researcher.research_knowledge_gap(gap)
                )
                
                if research_results['total_sources'] > 0:
                    # Enhanced RAG processing with user_id
                    enhancement_result = loop.run_until_complete(
                        orchestrator.rag_enhancer.enhance_rag_for_title(
                            title_id, 
                            research_results,
                            collection_name, 
                            merge_with_existing,
                            user_id  # PASS user_id HERE
                        )
                    )
                    
                    if enhancement_result['success']:
                        results = {
                            'status': 'success',
                            'title_id': title_id,
                            'user_id': user_id,
                            'sources_found': research_results['total_sources'],
                            'collection_name': enhancement_result['collection_name'],
                            'collection_strategy': enhancement_result.get('collection_strategy'),
                            'documents_added': enhancement_result['documents_added'],
                            'document_ids': enhancement_result.get('document_ids', []),
                            'research_summary': research_results['research_summary'],
                            'manual_suggestions_generated': enhancement_result.get('manual_suggestions_generated', 0),
                            'suggestions_breakdown': enhancement_result.get('suggestions_breakdown', {})
                        }
                        
                        # Add collection strategy info
                        results['collection_info'] = {
                            'final_collection_name': enhancement_result.get('collection_name'),
                            'collection_strategy': enhancement_result.get('collection_strategy'),
                            'merge_with_existing': merge_with_existing,
                            'requested_collection': collection_name,
                            'documents_added_to_lindex': True,
                            'content_type': 'gap_filler'
                        }
                        
                        return jsonify(results)
                    else:
                        return jsonify({
                            'status': 'enhancement_failed',
                            'error': enhancement_result.get('error')
                        })
                else:
                    return jsonify({
                        'status': 'no_sources',
                        'message': 'No relevant sources found'
                    })
                
            except Exception as e:
                logger.error(f"Error in enhance_single_title: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/manual_suggestions/<title_id>', methods=['POST'])
        def get_manual_suggestions(title_id):
            """Get manual action suggestions for a title - user_id in request body"""
            try:
                data = request.get_json() or {}
                user_id = data.get('user_id')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                # Query with user_id filter for RSA compatibility
                result = orchestrator.supabase.execute_query(
                    'GET', 
                    f'manual_action_suggestions?title_id=eq.{title_id}&user_id=eq.{user_id}&order=priority_score.desc'
                )
                
                if result['success']:
                    return jsonify({
                        "status": "success",
                        "title_id": title_id,
                        "user_id": user_id,
                        "suggestions": result['data'],
                        "total_suggestions": len(result['data'])
                    })
                else:
                    return jsonify({
                        "status": "error", 
                        "error": result.get('error', 'Failed to fetch suggestions')
                    }), 500
                
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/manual_suggestions/<title_id>/<suggestion_id>/status', methods=['PATCH'])
        def update_suggestion_status(title_id, suggestion_id):
            """Update status of a manual suggestion with user_id in request body"""
            try:
                data = request.get_json()
                new_status = data.get('status')
                user_id = data.get('user_id')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                if new_status not in ['suggested', 'in_progress', 'completed', 'rejected']:
                    return jsonify({"error": "Invalid status"}), 400
                
                update_data = {
                    "status": new_status,
                    "updated_at": datetime.now().isoformat()
                }
                
                if new_status == 'completed':
                    update_data["completed_at"] = datetime.now().isoformat()
                
                # Update with user_id filter for security
                result = orchestrator.supabase.execute_query(
                    'PATCH',
                    f'manual_action_suggestions?id=eq.{suggestion_id}&user_id=eq.{user_id}',
                    update_data
                )
                
                if result['success']:
                    return jsonify({
                        "status": "success", 
                        "updated": True,
                        "user_id": user_id,
                        "suggestion_id": suggestion_id,
                        "new_status": new_status
                    })
                else:
                    return jsonify({
                        "status": "error", 
                        "error": result.get('error', 'Failed to update suggestion')
                    }), 500
                
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/manual_suggestions', methods=['POST'])
        def get_all_user_suggestions():
            """Get all manual suggestions for a specific user - user_id in request body"""
            try:
                data = request.get_json() or {}
                user_id = data.get('user_id')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                # Optional filters
                status = data.get('status')
                action_type = data.get('action_type')
                priority_level = data.get('priority_level')
                limit = data.get('limit', 50)  # Default limit
                
                # Build query with user_id filter
                query = f'manual_action_suggestions?user_id=eq.{user_id}&order=priority_score.desc&limit={limit}'
                
                if status:
                    query += f'&status=eq.{status}'
                if action_type:
                    query += f'&action_type=eq.{action_type}'
                if priority_level:
                    query += f'&priority_level=eq.{priority_level}'
                
                result = orchestrator.supabase.execute_query('GET', query)
                
                if result['success']:
                    return jsonify({
                        "status": "success",
                        "user_id": user_id,
                        "suggestions": result['data'],
                        "total_suggestions": len(result['data']),
                        "filters": {
                            "status": status,
                            "action_type": action_type,
                            "priority_level": priority_level,
                            "limit": limit
                        }
                    })
                else:
                    return jsonify({
                        "status": "error", 
                        "error": result.get('error', 'Failed to fetch suggestions')
                    }), 500
                
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500

        @app.route('/manual_suggestions/<suggestion_id>', methods=['DELETE'])
        def delete_suggestion(suggestion_id):
            """Delete a manual suggestion - user_id in request body for security"""
            try:
                data = request.get_json() or {}
                user_id = data.get('user_id')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                # Delete with user_id filter for security
                result = orchestrator.supabase.execute_query(
                    'DELETE',
                    f'manual_action_suggestions?id=eq.{suggestion_id}&user_id=eq.{user_id}'
                )
                
                if result['success']:
                    return jsonify({
                        "status": "success", 
                        "deleted": True,
                        "suggestion_id": suggestion_id,
                        "user_id": user_id
                    })
                else:
                    return jsonify({
                        "status": "error", 
                        "error": result.get('error', 'Failed to delete suggestion')
                    }), 500
                
            except Exception as e:
                return jsonify({"status": "error", "error": str(e)}), 500


        @app.route('/job_status/<job_id>', methods=['GET'])
        def get_job_status(job_id):
            """Get status of an async job with detailed progress."""
            try:
                with job_lock:
                    job_info = jobs.get(job_id)
                
                if not job_info:
                    return jsonify({'error': 'Job not found'}), 404
                
                # Build response with all the detailed information
                response = {
                    "job_id": job_id,
                    "status": job_info.get('status', 'unknown'),
                    "progress": job_info.get('progress', 0),
                    "current_step": job_info.get('current_step', ''),
                    "steps": job_info.get('steps', [])[-5:],  # Last 5 steps
                    "phase": job_info.get('phase', 'unknown'),
                    "search_info": job_info.get('search_info', {
                        "strategy": '',
                        "llm_provider": '',
                        "embedding_dimension": 0,
                        "current_document": '',
                        "documents_being_searched": [],
                        "chunks_processed": 0,
                        "search_iterations": 0,
                        "max_iterations": 0,
                        "results_quality": 'unknown'
                    })
                }
                
                # Add result information if completed
                if job_info.get('status') == 'completed':
                    result = job_info.get('result', {})
                    response['result'] = result
                    
                    if isinstance(result, dict):
                        response['final_summary'] = {
                            "response_length": len(result.get('response', '')),
                            "chunks_used": result.get('chunks_used', 0),
                            "documents_searched": result.get('documents_searched', 0),
                            "method_used": result.get('method', 'unknown'),
                            "embedding_dimension": result.get('embedding_dimension', 0),
                            "time_seconds": result.get('time_seconds', 0),
                            "source_attribution": result.get('source_attribution', []),
                            "documents_used": result.get('documents_used', []),
                            "document_titles": result.get('document_titles', {})
                        }
                
                # Add error information if failed
                elif job_info.get('status') == 'failed':
                    response['error'] = job_info.get('error', 'Unknown error')
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error getting job status for {job_id}: {str(e)}")
                return jsonify({'error': f'Internal error: {str(e)}'}), 500
        
####################
        # Add this route to your main.py file in the create_app() function

        @app.route('/enhancement_details', methods=['GET'])
        def get_enhancement_details():
            """Get detailed enhancement information for knowledge-enhanced titles."""
            try:
                user_id = request.args.get('user_id')
                title_id = request.args.get('title_id')  # Optional - for specific title
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                logger.info(f"📋 Getting enhancement details for user: {user_id}, title: {title_id}")
                
                from knowledge_gap_http_supabase import HTTPSupabaseClient
                
                supabase_client = HTTPSupabaseClient()
                
                # Get knowledge-enhanced titles
                if title_id:
                    titles_query = f'Titles?id=eq.{title_id}&knowledge_enhanced=eq.true'
                else:
                    # Get all enhanced titles for user (adjust if your table has user_id column)
                    titles_query = 'Titles?knowledge_enhanced=eq.true'
                
                titles_response = supabase_client.execute_query('GET', titles_query)
                
                if not titles_response['success']:
                    return jsonify({
                        "status": "error",
                        "error": "Failed to fetch enhanced titles",
                        "details": titles_response.get('error')
                    }), 500
                
                enhanced_titles = titles_response['data']
                
                if not enhanced_titles:
                    return jsonify({
                        "status": "success",
                        "message": "No enhanced titles found",
                        "enhancements": [],
                        "user_id": user_id
                    })
                
                logger.info(f"📊 Found {len(enhanced_titles)} enhanced titles")
                
                # For each enhanced title, get the gap filler documents
                enhancement_details = []
                
                for title in enhanced_titles:
                    title_id = title.get('id')
                    title_name = title.get('Title', 'Unknown Title')
                    
                    logger.info(f"🔍 Getting gap filler documents for title: {title_name}")
                    
                    # Query gap filler documents for this title
                    # These are documents with gap_filler_info in processing_metadata
                    docs_query = f"lindex_documents?select=id,title,summary_short,summary_medium,source_type,processing_metadata,doc_size,chunk_count,url,author,importance_score&processing_status=eq.completed"
                    
                    docs_response = supabase_client.execute_query('GET', docs_query)
                    
                    if docs_response['success']:
                        all_docs = docs_response['data']
                        
                        # Filter for gap filler documents related to this title
                        gap_filler_docs = []
                        for doc in all_docs:
                            processing_metadata = doc.get('processing_metadata', {})
                            
                            # Check if this is a gap filler document for our title
                            if isinstance(processing_metadata, dict):
                                gap_filler_info = processing_metadata.get('gap_filler_info', {})
                                if gap_filler_info.get('original_title_id') == title_id:
                                    gap_filler_docs.append(doc)
                            elif isinstance(processing_metadata, str):
                                # Handle case where metadata is stored as JSON string
                                try:
                                    import json
                                    metadata_dict = json.loads(processing_metadata)
                                    gap_filler_info = metadata_dict.get('gap_filler_info', {})
                                    if gap_filler_info.get('original_title_id') == title_id:
                                        gap_filler_docs.append(doc)
                                except:
                                    continue
                        
                        logger.info(f"  📄 Found {len(gap_filler_docs)} gap filler documents")
                        
                        # Process the gap filler documents into enhancement details
                        enhancements_for_title = []
                        total_sources = len(gap_filler_docs)
                        total_documents_added = 0
                        source_types = {}
                        
                        for doc in gap_filler_docs:
                            processing_metadata = doc.get('processing_metadata', {})
                            
                            # Parse metadata if it's a string
                            if isinstance(processing_metadata, str):
                                try:
                                    processing_metadata = json.loads(processing_metadata)
                                except:
                                    processing_metadata = {}
                            
                            gap_filler_info = processing_metadata.get('gap_filler_info', {})
                            source_metadata = processing_metadata.get('source_metadata', {})
                            
                            source_type = gap_filler_info.get('data_source_type', doc.get('source_type', 'unknown'))
                            
                            # Count source types
                            source_types[source_type] = source_types.get(source_type, 0) + 1
                            total_documents_added += 1
                            
                            enhancement_item = {
                                "document_id": doc.get('id'),
                                "title": doc.get('title', 'Unknown Document'),
                                "source_type": source_type,
                                "source_description": self._get_source_type_description(source_type),
                                "summary": doc.get('summary_short') or doc.get('summary_medium', '')[:200],
                                "author": doc.get('author', 'Unknown'),
                                "url": doc.get('url', ''),
                                "doc_size": doc.get('doc_size', 0),
                                "chunk_count": doc.get('chunk_count', 0),
                                "importance_score": doc.get('importance_score', 0),
                                "research_timestamp": gap_filler_info.get('research_timestamp', ''),
                                "focus_keyword": gap_filler_info.get('focus_keyword', ''),
                                "original_source": gap_filler_info.get('original_source', ''),
                                "auto_generated": gap_filler_info.get('auto_generated', True)
                            }
                            
                            enhancements_for_title.append(enhancement_item)
                        
                        # Sort by importance score (highest first)
                        enhancements_for_title.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
                        
                        # Create summary for this title
                        title_enhancement = {
                            "title_id": title_id,
                            "title_name": title_name,
                            "status": title.get('status', 'Unknown'),
                            "knowledge_enhanced": title.get('knowledge_enhanced', False),
                            "gaps_closed": title.get('knowledge_gaps_closed', False),
                            "enhancement_timestamp": title.get('gap_closure_timestamp', ''),
                            "enhancement_method": title.get('gap_closure_method', ''),
                            "enhancement_summary": {
                                "total_sources": total_sources,
                                "documents_added": total_documents_added,
                                "source_types_breakdown": source_types,
                                "top_source_types": list(source_types.keys())[:3],
                                "research_quality": "high" if total_sources >= 5 else "medium" if total_sources >= 3 else "basic"
                            },
                            "enhancements": enhancements_for_title,
                            "actionable_items": self._create_actionable_items(enhancements_for_title, source_types)
                        }
                        
                        enhancement_details.append(title_enhancement)
                    
                    else:
                        logger.warning(f"⚠️ Failed to get documents for title {title_name}")
                
                # Create overall summary
                total_enhancements = sum(len(title['enhancements']) for title in enhancement_details)
                all_source_types = {}
                for title in enhancement_details:
                    for source_type, count in title['enhancement_summary']['source_types_breakdown'].items():
                        all_source_types[source_type] = all_source_types.get(source_type, 0) + count
                
                return jsonify({
                    "status": "success",
                    "user_id": user_id,
                    "query_timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total_enhanced_titles": len(enhanced_titles),
                        "total_enhancement_documents": total_enhancements,
                        "source_types_overall": all_source_types,
                        "most_common_sources": sorted(all_source_types.items(), key=lambda x: x[1], reverse=True)[:5]
                    },
                    "enhancement_details": enhancement_details
                })
                
            except Exception as e:
                logger.error(f"Error getting enhancement details: {e}")
                import traceback
                return jsonify({
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }), 500

        def _get_source_type_description(source_type):
            """Get user-friendly description for source types."""
            descriptions = {
                'academic_papers': '📚 Academic Research Papers - Peer-reviewed scholarly articles',
                'government_data': '🏛️ Government Data & Statistics - Official government sources',
                'industry_reports': '📊 Industry Reports - Professional market analysis and insights',
                'news_articles': '📰 News Articles - Current news and developments',
                'statistical_data': '📈 Statistical Data - Quantitative data and survey results',
                'textbooks': '📖 Educational Materials - Textbooks and learning resources',
                'market_data': '💹 Market Data - Market trends and analysis',
                'government_statistics': '📋 Government Statistics - Official statistical data',
                'investment_analysis': '💰 Investment Analysis - ROI and investment strategies',
                'technical_specifications': '🔧 Technical Specifications - Product details and specs',
                'financial_data': '💵 Financial Data - Cost analysis and financial information',
                'economic_indicators': '📊 Economic Indicators - Economic impact analysis'
            }
            return descriptions.get(source_type, f'📄 {source_type.replace("_", " ").title()} - Research source')

        def _create_actionable_items(enhancements, source_types):
            """Create actionable items based on the enhancements."""
            items = []
            
            if len(enhancements) > 0:
                items.append({
                    "action": "review_sources",
                    "title": "Review Enhanced Sources",
                    "description": f"Review {len(enhancements)} research sources added to enhance this content",
                    "priority": "high" if len(enhancements) >= 5 else "medium"
                })
            
            if 'academic_papers' in source_types:
                items.append({
                    "action": "cite_research",
                    "title": "Add Academic Citations", 
                    "description": f"Consider citing {source_types['academic_papers']} academic papers in your content",
                    "priority": "medium"
                })
            
            if 'statistical_data' in source_types or 'government_data' in source_types:
                items.append({
                    "action": "add_statistics",
                    "title": "Include Data & Statistics",
                    "description": "Incorporate statistical data and government sources for credibility",
                    "priority": "high"
                })
            
            if len(source_types) >= 3:
                items.append({
                    "action": "multi_source_synthesis",
                    "title": "Synthesize Multiple Sources",
                    "description": f"Content enhanced with {len(source_types)} different source types - ready for comprehensive writing",
                    "priority": "high"
                })
            
            return items

        # Also add a simplified endpoint for just getting enhancement counts
        @app.route('/enhancement_summary', methods=['GET'])
        def get_enhancement_summary():
            """Get a quick summary of enhancements for dashboard display."""
            try:
                user_id = request.args.get('user_id')
                
                if not user_id:
                    return jsonify({"status": "error", "error": "user_id is required"}), 400
                
                from knowledge_gap_http_supabase import HTTPSupabaseClient
                
                supabase_client = HTTPSupabaseClient()
                
                # Get all enhanced titles
                titles_response = supabase_client.execute_query('GET', 'Titles?knowledge_enhanced=eq.true')
                
                if not titles_response['success']:
                    return jsonify({"status": "error", "error": "Failed to fetch titles"}), 500
                
                enhanced_titles = titles_response['data']
                
                # Quick summary for dashboard
                summary_items = []
                
                for title in enhanced_titles:
                    summary_items.append({
                        "title_id": title.get('id'),
                        "title_name": title.get('Title', 'Unknown'),
                        "status": title.get('status', 'Unknown'),
                        "gaps_closed": title.get('knowledge_gaps_closed', False),
                        "enhancement_available": True,
                        "action_required": not title.get('knowledge_gaps_closed', False)
                    })
                
                return jsonify({
                    "status": "success",
                    "user_id": user_id,
                    "enhanced_titles_count": len(enhanced_titles),
                    "titles_with_enhancements": summary_items,
                    "query_timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting enhancement summary: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500
####################


#####################        
        @app.route('/test_agent', methods=['GET'])
        def test_agent_endpoint():
            """Test endpoint to verify agent functionality."""
            try:
                test_query = "What are the key things to consider before buying a house?"
                test_collection = "rag_house_and_real_estate"  # Use your actual collection name
                
                logger.info(f"🧪 Testing agent with query: {test_query}")
                
                engine = get_rag_engine(test_collection)
                
                # Test simple query first
                simple_result = engine.query_simple(test_query, num_results=3)
                
                # Test fixed agentic query if agents are available
                agentic_result = None
                if AGENTS_AVAILABLE:
                    agentic_result = engine.query_agentic_fixed(test_query, max_docs=2)
                
                return jsonify({
                    "test_query": test_query,
                    "collection": test_collection,
                    "agents_available": AGENTS_AVAILABLE,
                    "embedding_dimension": embedding_manager.get_dimension(),
                    "simple_result": {
                        "status": simple_result.get('status'),
                        "chunks_used": simple_result.get('chunks_used', 0),
                        "documents_searched": simple_result.get('documents_searched', 0),
                        "response_length": len(simple_result.get('response', ''))
                    },
                    "agentic_result": {
                        "status": agentic_result.get('status') if agentic_result else "not_tested",
                        "chunks_used": agentic_result.get('chunks_used', 0) if agentic_result else 0,
                        "documents_searched": agentic_result.get('documents_searched', 0) if agentic_result else 0,
                        "response_length": len(agentic_result.get('response', '')) if agentic_result else 0
                    } if agentic_result else {"status": "agents_not_available"}
                })
                
            except Exception as e:
                logger.error(f"Test agent error: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        logger.info("✅ Flask application created successfully")
        if KNOWLEDGE_GAP_AVAILABLE:
            try:
                # Ensure dependencies are available
                logger.info("🔧 Making dependencies available for Knowledge Gap Filler...")
                logger.info(f"   - document_processor: {type(document_processor).__name__}")
                logger.info(f"   - background_executor: {type(background_executor).__name__}")
                
                log_memory_usage()
                
                # Integrate with full dependencies
                integrate_knowledge_gap_filler_http(app)
                logger.info("✅ HTTP-based Knowledge Gap Filler integrated successfully")
                
                log_memory_usage()
                
            except Exception as e:
                logger.error(f"❌ Failed to integrate Knowledge Gap Filler: {e}")
                import traceback
                logger.error(f"   Full traceback: {traceback.format_exc()}")
                # Force garbage collection on error
                gc.collect()
        else:
            logger.warning("⚠️ Knowledge Gap Filler not available - routes not added")

        return app
        
    except Exception as e:
        logger.error(f"❌ Failed to create Flask application: {str(e)}")
        return None

if __name__ == '__main__':
    try:
        app = create_app()
        if app:
            logger.info("🚀 Starting Flask application...")
            # Add memory monitoring before starting
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"🧠 Final memory usage before start: {memory_mb:.1f} MB")
            
            # Start with conservative settings
            app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), threaded=True)
        else:
            logger.error("❌ Failed to create Flask application")
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")