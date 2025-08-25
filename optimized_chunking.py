from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import TokenTextSplitter, SemanticSplitterNodeParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class OptimizedDocumentChunker:
    """
    Enhanced document chunking implementation with better performance.
    """
    
    def __init__(self, embed_model, batch_size=8, max_workers=4):
        """Initialize the chunker with the given embedding model."""
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.max_workers = max_workers
        
    def chunk_document(self, text: str, docid: str, metadata: Optional[Dict] = None, source_type: str = "document") -> List[TextNode]:
        """
        Optimized document chunking strategy that adapts to document content.
        
        Args:
            text: Document text content
            docid: Document identifier
            metadata: Optional custom metadata
            source_type: Source type of the document (e.g., 'pdf', 'webpage', 'email')
        
        Returns:
            List of TextNode objects
        """
        start_time = time.time()
        logger.info(f"Starting chunking for document {docid}")
        
        # Basic content analysis for chunking strategy
        is_markdown = self._detect_markdown(text)
        has_long_paragraphs = len(re.findall(r'\n\n.{1000,}', text)) > 3
        
        # Select chunking strategy based on document analysis
        if is_markdown:
            chunks = self._chunk_markdown(text, docid)
        elif has_long_paragraphs:
            chunks = self._chunk_semantic(text, docid)
        else:
            chunks = self._chunk_basic(text, docid)
            
        # Add document metadata to each chunk
        for chunk in chunks:
            # Add base metadata
            chunk.metadata.update({
                'docid': str(docid),
                'source_type': source_type,
                'chunk_count': len(chunks),
                'created_at': datetime.now().isoformat()
            })
            
            # Add custom metadata if provided
            if metadata:
                chunk.metadata.update(metadata)
        
        chunk_time = time.time() - start_time
        logger.info(f"Chunking completed in {chunk_time:.2f}s - {len(chunks)} chunks created")
        
        return chunks
    
    def _detect_markdown(self, text: str) -> bool:
        """Detect if text appears to be markdown formatted."""
        # Look for markdown indicators
        md_patterns = [
            r'^#+\s+.+$',  # Headers
            r'^\s*[*-]\s+.+$',  # List items
            r'^\s*\d+\.\s+.+$',  # Numbered lists
            r'\[.+\]\(.+\)',  # Links
            r'!\[.+\]\(.+\)',  # Images
            r'```\w*\n[\s\S]*?\n```',  # Code blocks
            r'\*\*.+\*\*',  # Bold
            r'_.+_'  # Italic
        ]
        
        # Check for markdown patterns
        md_count = sum(1 for pattern in md_patterns if re.search(pattern, text, re.MULTILINE))
        return md_count >= 3  # Consider markdown if 3+ patterns found
    
    def _chunk_markdown(self, text: str, docid: str) -> List[TextNode]:
        """Markdown-optimized chunking strategy."""
        from llama_index.core.node_parser import MarkdownNodeParser
        
        try:
            # Use MarkdownNodeParser for better handling of headings and structure
            parser = MarkdownNodeParser(include_metadata=True, include_heading_metadata=True)
            doc = Document(text=text)
            nodes = parser.get_nodes_from_documents([doc])
            
            # Process nodes to ensure minimal size
            processed_nodes = []
            current_text = ""
            current_metadata = {'source': f'doc_{docid}'}
            
            for i, node in enumerate(nodes):
                node_text = node.get_content()
                node_metadata = node.metadata.copy() if hasattr(node, 'metadata') else {}
                
                # If node is too small, accumulate
                if len(node_text) < 200 and i < len(nodes) - 1:
                    current_text += node_text + "\n\n"
                    # Keep heading metadata if available
                    if 'heading' in node_metadata:
                        current_metadata['heading'] = node_metadata['heading']
                else:
                    # Add accumulated text to current node if any
                    if current_text:
                        node_text = current_text + node_text
                        current_text = ""
                        node_metadata.update(current_metadata)
                    
                    # Create TextNode with processed content
                    processed_node = TextNode(
                        text=node_text,
                        metadata=node_metadata,
                        id_=f"doc_{docid}_chunk_{i}"
                    )
                    processed_nodes.append(processed_node)
            
            # Handle any remaining accumulated text
            if current_text:
                processed_nodes.append(TextNode(
                    text=current_text,
                    metadata=current_metadata,
                    id_=f"doc_{docid}_chunk_{len(processed_nodes)}"
                ))
            
            return processed_nodes
            
        except Exception as e:
            logger.error(f"Markdown chunking failed: {str(e)}")
            # Fall back to basic chunking
            return self._chunk_basic(text, docid)
    
    def _chunk_semantic(self, text: str, docid: str) -> List[TextNode]:
        """Semantically-aware chunking strategy for better context preservation."""
        try:
            # Use semantic splitter for better context splits
            semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model
            )
            
            doc = Document(text=text)
            nodes = semantic_splitter.get_nodes_from_documents([doc])
            
            # Create proper TextNodes
            text_nodes = []
            for i, node in enumerate(nodes):
                text_nodes.append(TextNode(
                    text=node.get_content(),
                    id_=f"doc_{docid}_chunk_{i}",
                    metadata={'chunk_type': 'semantic', 'source': f'doc_{docid}'}
                ))
                
            return text_nodes
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}")
            # Fall back to basic chunking
            return self._chunk_basic(text, docid)
    
    def _chunk_basic(self, text: str, docid: str) -> List[TextNode]:
        """Basic token-based chunking with overlap."""
        try:
            # Use token splitter with separator awareness
            splitter = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator="\n\n"  # Paragraph-aware splitting
            )
            
            # Get nodes
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
            
            # Create proper TextNodes
            text_nodes = []
            for i, node in enumerate(nodes):
                text_nodes.append(TextNode(
                    text=node.get_content(),
                    id_=f"doc_{docid}_chunk_{i}",
                    metadata={'chunk_type': 'token', 'source': f'doc_{docid}'}
                ))
                
            return text_nodes
            
        except Exception as e:
            logger.error(f"Basic chunking failed: {str(e)}")
            # Manual chunking as last resort
            chunks = self._manual_chunk(text, docid)
            return chunks
    
    def _manual_chunk(self, text: str, docid: str) -> List[TextNode]:
        """Last-resort manual paragraph-based chunking."""
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed ~500 tokens
            if len(current_chunk) + len(para) > 2000:
                # Store current chunk
                if current_chunk:
                    chunks.append(TextNode(
                        text=current_chunk,
                        id_=f"doc_{docid}_chunk_{len(chunks)}",
                        metadata={'chunk_type': 'manual', 'source': f'doc_{docid}'}
                    ))
                    current_chunk = para
                else:
                    # If a single paragraph is too large, split it
                    if len(para) > 2000:
                        # Split large paragraph into ~1500 char chunks
                        for j in range(0, len(para), 1500):
                            chunk_text = para[j:j+1500]
                            chunks.append(TextNode(
                                text=chunk_text,
                                id_=f"doc_{docid}_chunk_{len(chunks)}",
                                metadata={'chunk_type': 'manual_split', 'source': f'doc_{docid}'}
                            ))
                    else:
                        current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(TextNode(
                text=current_chunk,
                id_=f"doc_{docid}_chunk_{len(chunks)}",
                metadata={'chunk_type': 'manual', 'source': f'doc_{docid}'}
            ))
            
        return chunks
    
    def batch_process(self, nodes: List[TextNode], docid: str) -> List[TextNode]:
        """Process nodes in parallel batches for embedding."""
        start_time = time.time()
        processed_nodes = []
        
        # Process in batches
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i + self.batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_node, node, docid, i + j): j 
                    for j, node in enumerate(batch)
                }
                
                for future in as_completed(futures):
                    try:
                        node = future.result(timeout=30)  # 30-second timeout per node
                        if node:
                            processed_nodes.append(node)
                    except Exception as e:
                        logger.error(f"Node processing failed: {str(e)}")
        
        logger.info(f"Batch processing completed in {time.time() - start_time:.2f}s - {len(processed_nodes)}/{len(nodes)} nodes processed")
        return processed_nodes
    
    def _process_node(self, node: TextNode, docid: str, index: int, source_type: str = "document") -> Optional[TextNode]:
        """
        Process a single node to add embedding and validate content.
        
        Args:
            node: TextNode to process
            docid: Document identifier
            index: Node index within document
            source_type: Source type of the document
            
        Returns:
            Processed TextNode or None if invalid
        """
        try:
            # Get text content
            text = node.get_content()
            
            # Skip empty or tiny nodes
            if not text or len(text) < 20:
                return None
                
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(text)
            
            # Set embedding and enhance metadata
            node.embedding = embedding
            
            # Enhance metadata
            if not hasattr(node, 'metadata') or not node.metadata:
                node.metadata = {}
                
            # Add standardized metadata
            node.metadata.update({
                'docid': str(docid),
                'source_type': source_type,
                'chunk_index': index,
                'text': text,  # Store full text for retrieval
                'text_sample': text[:200],  # Preview for debugging
                'content_length': len(text),
                'created_at': datetime.now().isoformat() 
            })
            
            return node
            
        except Exception as e:
            logger.error(f"Node processing error: {str(e)}")
            return None