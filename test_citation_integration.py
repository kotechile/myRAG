
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

# Mock the imports BEFORE importing the module
sys.modules['llama_index.core.schema'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.vector_stores.supabase'] = MagicMock()
sys.modules['llama_index.vector_stores.supabase.SupabaseVectorStore'] = MagicMock()
sys.modules['llama_index.core.VectorStoreIndex'] = MagicMock()
sys.modules['llama_index.core.StorageContext'] = MagicMock()

# Import the modified class
# We need to use patch.dict to ensure these mocks persist during import if they are used at top level
from optimized_embedding import BatchDocumentProcessor

class TestCitationTransfer(unittest.TestCase):
    def setUp(self):
        self.mock_chunker = MagicMock()
        self.mock_embedding_pipeline = MagicMock()
        self.mock_supabase = MagicMock()
        
        # Setup mock behavior
        self.mock_chunker.chunk_document.return_value = ["chunk1", "chunk2"]
        # Mock embedding pipeline to return a dimension
        self.mock_embedding_pipeline._get_embedding_dimension.return_value = 1536
        self.mock_embedding_pipeline.process_document.return_value = {
            'success': True, 
            'total_nodes': 2,
            'successful_nodes': 2,
            'failed_nodes': 0
        }
        
        self.processor = BatchDocumentProcessor(
            chunker=self.mock_chunker,
            embedding_pipeline=self.mock_embedding_pipeline,
            supabase_client=self.mock_supabase
        )
        
        # Mock _fetch_document to return some text
        self.processor._fetch_document = MagicMock(return_value="Some document text")
        self.processor._generate_summary = MagicMock(return_value={})
        self.processor._update_document_status = MagicMock()

    def test_process_document_passes_metadata(self):
        """Test that extra_metadata is passed to chunker"""
        docid = "test_doc_123"
        collection_name = "test_collection"
        source_type = "pdf"
        citations = {"citations": [{"url": "http://example.com", "title": "Example"}]}
        
        # Call process_document with extra_metadata
        self.processor.process_document(
            docid=docid,
            collection_name=collection_name,
            source_type=source_type,
            extra_metadata=citations
        )
        
        # Verify chunker was called with merged metadata
        expected_metadata = {
            'collection_name': collection_name,
            'source_type': source_type,
            **citations
        }
        
        self.mock_chunker.chunk_document.assert_called_once()
        # Verify arguments - note that chunk_document call signature might be different depending on how it's called
        # The key is checking if the metadata dict contains our citations
        call_args = self.mock_chunker.chunk_document.call_args
        _, kwargs = call_args
        
        # Check positional or keyword args
        if kwargs.get('metadata'):
            metadata_arg = kwargs['metadata']
        else:
            # Maybe it was passed positionally? Let's check call_args[0] if needed but keyword is safer
            # Based on the code: self.chunker.chunk_document(text=..., docid=..., metadata=..., source_type=...)
            metadata_arg = kwargs.get('metadata')
            
        # If not found in kwargs, try checking if it was passed via named arguments in call
        if not metadata_arg:
             # This handles if it's called with positional args, which it isn't in my code currently but good for robust test
             pass

        print(f"Captured metadata: {metadata_arg}")
        
        self.assertEqual(kwargs['text'], "Some document text")
        self.assertEqual(kwargs['docid'], docid)
        self.assertEqual(metadata_arg, expected_metadata)
        self.assertEqual(kwargs['source_type'], source_type)
        
        print("✅ BatchDocumentProcessor unit test passed: extra_metadata correctly passed to chunker")

if __name__ == '__main__':
    unittest.main()
