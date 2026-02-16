# Fixed optimized_embedding.py - with dynamic dimension support

from llama_index.core.schema import TextNode, Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from datetime import datetime
import json
import gc
import traceback

logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """
    Optimized embedding and storage pipeline for document processing.
    """
    
    def __init__(self, embed_model, db_connection: str, batch_size: int = 10):
        """Initialize the pipeline with the given embedding model."""
        self.embed_model = embed_model
        self.db_connection = db_connection
        self.batch_size = batch_size
        self._vector_stores = {}  # Cache for vector stores
        
    def _get_embedding_dimension(self) -> int:
        """Get the current embedding dimension dynamically."""
        try:
            # Check if it's our optimized model with fitted optimizer
            if hasattr(self.embed_model, 'optimizer') and self.embed_model.optimizer and self.embed_model.optimizer.fitted:
                dim = self.embed_model.optimizer.target_dim
                logger.info(f"📏 Pipeline using fitted optimizer dimension: {dim}")
                return dim
            # Check if it has dimension property
            elif hasattr(self.embed_model, 'dimension'):
                dim = self.embed_model.dimension
                logger.info(f"📏 Pipeline using model dimension property: {dim}")
                return dim
            else:
                # Test with a small text to get the dimension
                logger.info("📏 Pipeline testing embedding to determine dimension...")
                test_embedding = self.embed_model.get_text_embedding("test")
                dim = len(test_embedding)
                logger.info(f"📏 Pipeline determined dimension by testing: {dim}")
                return dim
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension in pipeline: {e}")
            # Default fallback
            return 1536
        
    def process_document(self, nodes: List[TextNode], docid: str, 
                         collection_name: str) -> Dict[str, Any]:
        """
        Process document nodes with optimized embedding and storage.
        
        Args:
            nodes: List of TextNodes to process
            docid: Document ID
            collection_name: Vector store collection name
            
        Returns:
            Dict with processing statistics
        """
        start_time = time.time()
        logger.info(f"Processing document {docid} with {len(nodes)} nodes")
        logger.info(f"Current embedding dimension: {self._get_embedding_dimension()}")
        
        results = {
            'docid': docid,
            'collection_name': collection_name,
            'total_nodes': len(nodes),
            'successful_nodes': 0,
            'failed_nodes': 0,
            'storage_errors': 0,
            'embedding_time': 0,
            'storage_time': 0,
            'node_sizes': [],
            'embedding_dimension': self._get_embedding_dimension()
        }
        
        # Skip if no nodes
        if not nodes:
            logger.warning(f"No nodes to process for document {docid}")
            return results
            
        # 1. Generate embeddings in batches
        embedding_start = time.time()
        processed_nodes = self._generate_embeddings(nodes)
        results['embedding_time'] = time.time() - embedding_start
        
        # Update stats from embedding generation
        results['successful_nodes'] = len(processed_nodes)
        results['failed_nodes'] = len(nodes) - len(processed_nodes)
        results['node_sizes'] = [len(node.get_content()) for node in processed_nodes]
        
        # If all nodes failed, return early
        if not processed_nodes:
            logger.error(f"All nodes failed embedding for document {docid}")
            return results
            
        # 2. Store nodes in vector store
        storage_start = time.time()
        store_results = self._store_nodes(processed_nodes, collection_name)
        results['storage_time'] = time.time() - storage_start
        
        # Update storage results
        results['storage_errors'] = store_results.get('errors', 0)
        
        # 3. Run garbage collection to free memory
        gc.collect()
        
        # Log results
        total_time = time.time() - start_time
        logger.info(
            f"Document {docid} processed in {total_time:.2f}s: "
            f"{results['successful_nodes']}/{len(nodes)} nodes, "
            f"{results['storage_errors']} storage errors, "
            f"dimension: {results['embedding_dimension']}"
        )
        
        return results
        
    def _generate_embeddings(self, nodes: List[TextNode]) -> List[TextNode]:
        """Generate embeddings for nodes in parallel batches."""
        processed_nodes = []
        
        # Process in batches to manage memory usage
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            processed_nodes.extend(batch_results)
            
        return processed_nodes
            
    def _process_batch(self, nodes: List[TextNode]) -> List[TextNode]:
        """Process a batch of nodes with parallel embedding generation."""
        processed_batch = []
        
        try:
            # Create embedding requests in parallel
            with ThreadPoolExecutor(max_workers=min(5, len(nodes))) as executor:
                futures = {
                    executor.submit(self._embed_node, node): node 
                    for node in nodes
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=20)  # 20-second timeout
                        if result:
                            processed_batch.append(result)
                    except Exception as e:
                        logger.error(f"Node embedding failed: {str(e)}")
                        
            return processed_batch
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return processed_batch
    
    def _embed_node(self, node: TextNode) -> Optional[TextNode]:
        """Generate embedding for a single node."""
        try:
            # Get node text
            text = node.get_content()
            
            # Skip if too short
            if not text or len(text) < 20:
                logger.warning(f"Skipping short node: {len(text) if text else 0} chars")
                return None
                
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(text)
            
            # Set embedding
            node.embedding = embedding
            
            # Ensure metadata exists
            if not hasattr(node, 'metadata') or not node.metadata:
                node.metadata = {}
                
            # Add text to metadata for retrieval
            if 'text' not in node.metadata:
                node.metadata['text'] = text
                
            return node
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return None
            
    def _store_nodes(self, nodes: List[TextNode], collection_name: str) -> Dict[str, Any]:
        """Store nodes in vector store with optimized batching."""
        results = {'stored': 0, 'errors': 0}
        
        # Skip if no nodes
        if not nodes:
            return results
            
        try:
            # Get or create vector store
            vector_store = self._get_vector_store(collection_name)
            
            # Process in smaller batches for reliable storage
            for i in range(0, len(nodes), 20):  # Store 20 at a time
                batch = nodes[i:i + 20]
                
                try:
                    # Store batch
                    vector_store.add(batch)
                    results['stored'] += len(batch)
                    
                except Exception as e:
                    logger.error(f"Batch storage error: {str(e)}")
                    results['errors'] += len(batch)
                    
                    # Try one by one as fallback
                    for node in batch:
                        try:
                            vector_store.add([node])
                            results['stored'] += 1
                            results['errors'] -= 1
                        except Exception as node_error:
                            logger.error(f"Individual node storage error: {str(node_error)}")
                
        except Exception as e:
            logger.error(f"Vector store error: {str(e)}")
            results['errors'] += len(nodes)
            
        return results
            
    def _get_vector_store(self, collection_name: str) -> SupabaseVectorStore:
        """Get or create vector store for the given collection with dynamic dimension."""
        # Create cache key that includes dimension for proper caching
        dimension = self._get_embedding_dimension()
        cache_key = f"{collection_name}_{dimension}"
        
        # Check cache first
        if cache_key in self._vector_stores:
            logger.info(f"📁 Using cached vector store for {collection_name} (dim: {dimension})")
            return self._vector_stores[cache_key]
            
        try:
            # Try to connect to existing vector store first
            logger.info(f"🔍 Attempting to connect to existing vector store: {collection_name}")
            
            # Try to create vector store with compatibility handling
            try:
                vector_store = SupabaseVectorStore(
                    postgres_connection_string=self.db_connection,
                    collection_name=collection_name,
                    dimension=dimension  # Use dynamic dimension
                )
                logger.info(f"✅ Successfully connected to existing vector store: {collection_name}")
                
            except Exception as compat_error:
                error_str = str(compat_error)
                
                # Handle PostgreSQL constraint violations - collection already exists
                if "pg_type_typname_nsp_index" in error_str or "duplicate key value violates unique constraint" in error_str:
                    logger.info(f"📋 Collection '{collection_name}' already exists, connecting to existing collection")
                    # Try without dimension parameter to connect to existing collection
                    try:
                        vector_store = SupabaseVectorStore(
                            postgres_connection_string=self.db_connection,
                            collection_name=collection_name
                        )
                        logger.info(f"✅ Successfully connected to existing collection: {collection_name}")
                    except Exception as connect_error:
                        logger.error(f"Failed to connect to existing collection {collection_name}: {str(connect_error)}")
                        raise connect_error
                        
                elif "http_client" in error_str or "SyncPostgrestClient" in error_str:
                    logger.warning(f"⚠️ Supabase compatibility issue detected, trying alternative approach")
                    # Try without dimension parameter as fallback
                    vector_store = SupabaseVectorStore(
                        postgres_connection_string=self.db_connection,
                        collection_name=collection_name
                    )
                else:
                    raise compat_error
            
            # Cache for future use with dimension-specific key
            self._vector_stores[cache_key] = vector_store
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Vector store connection error for {collection_name}: {str(e)}")
            raise


class BatchDocumentProcessor:
    """
    Processor for handling multiple documents in batches.
    """
    
    def __init__(self, chunker, embedding_pipeline, supabase_client, max_workers=3):
        """Initialize the batch processor."""
        self.chunker = chunker
        self.embedding_pipeline = embedding_pipeline
        self.supabase_client = supabase_client
        self.max_workers = max_workers
        
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from the pipeline."""
        return self.embedding_pipeline._get_embedding_dimension()
        
    def process_documents(self, doc_ids: List[str], collection_name: str, source_type: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process multiple documents in parallel batches.
        
        Args:
            doc_ids: List of document IDs to process
            collection_name: Name of the vector collection
            source_type: Optional dictionary mapping document IDs to their source types
                        e.g., {'doc123': 'pdf', 'doc456': 'email'}
        
        Returns:
            Dict with processing results
        """
        start_time = time.time()
        results = {
            'total_documents': len(doc_ids),
            'successful': 0,
            'failed': 0,
            'documents': {}
        }
        
        # Initialize source_type dict if not provided
        if source_type is None:
            source_type = {}
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit processing tasks with source_type if available
            for docid in doc_ids:
                doc_source_type = source_type.get(docid, "document")  # Default to 'document' if not specified
                
                # Submit task with source_type
                future = executor.submit(
                    self.process_document, 
                    docid, 
                    collection_name,
                    doc_source_type
                )
                futures[future] = docid
            
            # Process results as they complete
            for future in as_completed(futures):
                docid = futures[future]
                try:
                    doc_result = future.result(timeout=300)  # 5-minute timeout per document
                    
                    # Update results
                    if doc_result.get('success', False):
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        
                    results['documents'][docid] = doc_result
                    
                except Exception as e:
                    logger.error(f"Document {docid} processing failed: {str(e)}")
                    results['failed'] += 1
                    results['documents'][docid] = {
                        'docid': docid,
                        'success': False,
                        'error': str(e)
                    }
        
        # Log summary
        total_time = time.time() - start_time
        logger.info(
            f"Batch processing completed in {total_time:.2f}s: "
            f"{results['successful']}/{len(doc_ids)} documents, "
            f"{results['failed']} failed"
        )
        
        return results
        
    def process_document(self, docid: str, collection_name: str, source_type: str = "document", extra_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            docid: Document ID
            collection_name: Collection name for vector storage
            source_type: Source type of the document (e.g., 'pdf', 'webpage', 'email')
            extra_metadata: Optional dictionary of additional metadata (e.g., citations)
            
        Returns:
            Dict with processing results
        """
        start_time = time.time()
        logger.info(f"Processing document {docid} with embedding dimension: {self._get_embedding_dimension()}")
        result = {
            'docid': docid,
            'collection_name': collection_name,
            'source_type': source_type,
            'success': False,
            'chunking': {},
            'embedding': {},
            'summary': {},
            'embedding_dimension': self._get_embedding_dimension()
        }
        
        try:
            # 1. Fetch document content
            doc_content = self._fetch_document(docid)
            
            if not doc_content:
                result['error'] = "Document content not found"
                return result
                
            # 2. Chunk document with source_type and extra_metadata
            chunk_metadata = {'collection_name': collection_name, 'source_type': source_type}
            if extra_metadata:
                chunk_metadata.update(extra_metadata)
                
            nodes = self.chunker.chunk_document(
                text=doc_content,
                docid=docid,
                metadata=chunk_metadata,
                source_type=source_type  # Pass source_type to chunker
            )
            
            result['chunking'] = {
                'total_chunks': len(nodes),
                'time': time.time() - start_time
            }
            
            if not nodes:
                result['error'] = "Chunking failed - no chunks created"
                return result
                
            # 3. Process embedding and storage
            embedding_result = self.embedding_pipeline.process_document(
                nodes=nodes,
                docid=docid,
                collection_name=collection_name
            )
            
            result['embedding'] = embedding_result
            
            # 4. Generate summary
            summary_result = self._generate_summary(docid, collection_name)
            result['summary'] = summary_result
            
            # 5. Update document status with source_type
            self._update_document_status(docid, embedding_result, summary_result, source_type)
            
            # Set success flag
            result['success'] = True
            result['total_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing error for {docid}: {str(e)}")
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            return result
                
    def _fetch_document(self, docid: str) -> Optional[str]:
        """Fetch document content from storage."""
        try:
            # Check if we're using HTTPSupabaseClient
            if hasattr(self.supabase_client, 'execute_query'):
                # Use HTTP client method
                response = self.supabase_client.execute_query(
                    'GET',
                    f'lindex_documents?select=parsedText&id=eq.{docid}'
                )
                
                if response.get('success') and response.get('data') and response['data']:
                    return response['data'][0].get('parsedText')
            else:
                # Use regular Supabase client method
                response = self.supabase_client.table("lindex_documents").select(
                    "parsedText"
                ).eq("id", docid).execute()
                
                if response.data and response.data[0].get('parsedText'):
                    return response.data[0]['parsedText']
                
            return None
            
        except Exception as e:
            logger.error(f"Error fetching document {docid}: {str(e)}")
            return None
        
    def _generate_summary(self, docid: str, collection_name: str) -> Dict[str, Any]:
        """Generate document summary from chunks."""
        # This is a placeholder - implement your actual summary generation
        # You can call your existing summary function here
        return {'generated': False, 'message': 'Summary generation not implemented'}
            
    def _update_document_status(self, docid: str, embedding_result: Dict, 
                          summary_result: Dict, source_type: str = "document") -> None:
        """
        Update document status in database.
        
        Args:
            docid: Document ID
            embedding_result: Results from embedding process
            summary_result: Results from summary generation
            source_type: Source type of the document
        """
        try:
            # Prepare update data
            update_data = {
                "last_processed": datetime.now().isoformat(),
                "processing_status": "completed",
                "in_vector_store": True,
                "source_type": source_type,  # Store source_type at document level
                "embedding_dimension": self._get_embedding_dimension(),  # Store dimension info
                "chunk_stats": {
                    "total_chunks": embedding_result.get('total_nodes', 0),
                    "processed_chunks": embedding_result.get('successful_nodes', 0),
                    "failed_chunks": embedding_result.get('failed_nodes', 0),
                    "embedding_dimension": embedding_result.get('embedding_dimension', 'unknown')
                }
            }
            
            # Add summary if available
            if summary_result.get('summary_short'):
                update_data["summary_short"] = summary_result['summary_short']
                
            if summary_result.get('summary_medium'):
                update_data["summary_medium"] = summary_result['summary_medium']
                
            # Update document record (remove embedding_dimension if it doesn't exist in schema)
            # Remove embedding_dimension from update_data to avoid schema errors
            if "embedding_dimension" in update_data:
                del update_data["embedding_dimension"]
            
            self.supabase_client.table("lindex_documents").update(
                update_data
            ).eq("id", docid).execute()
            
        except Exception as e:
            logger.error(f"Failed to update document status for {docid}: {str(e)}")