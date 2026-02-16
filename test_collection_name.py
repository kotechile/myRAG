#!/usr/bin/env python3
"""
Test script to check collection name matching.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_rag_engine

def test_collection_names():
    """Test different collection names to see which one works."""
    
    print("🔍 Testing Collection Name Matching")
    print("=" * 50)
    
    # Known collection name from debug
    known_collection = "rag_house_and_real_estate_128D_OPT"
    
    # Test different variations
    test_names = [
        known_collection,
        "rag_house_and_real_estate",
        "house_and_real_estate",
        "real_estate",
        "rag_house",
        "house",
        "real_estate_128D_OPT",
        "rag_house_and_real_estate_128D",
        "rag_house_and_real_estate_128D_OPT",
        "RAG_HOUSE_AND_REAL_ESTATE_128D_OPT",  # Different case
        "rag_house_and_real_estate_128d_opt",  # Lowercase
    ]
    
    for collection_name in test_names:
        print(f"\n🧪 Testing collection name: '{collection_name}'")
        try:
            engine = get_rag_engine(collection_name)
            metadata_count = len(engine.document_metadata)
            print(f"   ✅ Success! Loaded {metadata_count} documents")
            
            if metadata_count > 0:
                print(f"   📚 Sample documents:")
                for i, (doc_id, doc_meta) in enumerate(list(engine.document_metadata.items())[:3]):
                    print(f"      - {doc_id}: {doc_meta.get('title', 'No title')}")
                
                if metadata_count > 3:
                    print(f"      ... and {metadata_count - 3} more")
            else:
                print(f"   ⚠️ No metadata loaded")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n✅ Test complete!")
    print(f"💡 Use the collection name that shows the most documents: '{known_collection}'")

if __name__ == "__main__":
    test_collection_names()

