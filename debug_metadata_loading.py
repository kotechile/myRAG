#!/usr/bin/env python3
"""
Debug script to diagnose metadata loading issues.
This will help identify why "Available metadata keys: []" appears in logs.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge_gap_http_supabase import HTTPSupabaseClient

def debug_metadata_loading():
    """Debug the metadata loading process step by step."""
    
    print("🔍 Debugging Metadata Loading Issues")
    print("=" * 50)
    
    try:
        # Initialize Supabase client
        print("1. Initializing Supabase client...")
        supabase = HTTPSupabaseClient()
        print("✅ Supabase client initialized")
        
        # Check if we can connect to the database
        print("\n2. Testing database connection...")
        try:
            # Simple query to test connection
            test_response = supabase.table("lindex_documents").select("id").limit(1).execute()
            print(f"✅ Database connection successful. Found {len(test_response.data)} documents total")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return
        
        # Check all collections
        print("\n3. Checking available collections...")
        try:
            collections_response = supabase.table("lindex_collections").select("id, name").execute()
            collections = collections_response.data
            print(f"✅ Found {len(collections)} collections:")
            for col in collections:
                print(f"   - ID: {col['id']}, Name: '{col['name']}'")
        except Exception as e:
            print(f"❌ Failed to load collections: {e}")
            return
        
        # Check documents in vector store
        print("\n4. Checking documents in vector store...")
        try:
            vector_docs_response = supabase.table("lindex_documents").select(
                "id, title, in_vector_store, collectionId, source_type"
            ).eq("in_vector_store", True).execute()
            
            vector_docs = vector_docs_response.data
            print(f"✅ Found {len(vector_docs)} documents in vector store")
            
            if vector_docs:
                print("   Sample documents:")
                for doc in vector_docs[:5]:
                    print(f"   - ID: {doc['id']}, Title: '{doc.get('title', 'No title')}', Collection: {doc.get('collectionId')}, Type: {doc.get('source_type', 'unknown')}")
            else:
                print("   ⚠️ No documents found in vector store!")
                return
                
        except Exception as e:
            print(f"❌ Failed to load vector store documents: {e}")
            return
        
        # Check documents by collection
        print("\n5. Checking documents by collection...")
        collection_map = {str(col["id"]): col["name"] for col in collections}
        
        for col_id, col_name in collection_map.items():
            try:
                col_docs = [doc for doc in vector_docs if str(doc.get("collectionId")) == col_id]
                print(f"   Collection '{col_name}' (ID: {col_id}): {len(col_docs)} documents")
                
                if col_docs:
                    print(f"      Sample: {col_docs[0].get('title', 'No title')} (ID: {col_docs[0]['id']})")
            except Exception as e:
                print(f"   ❌ Error checking collection {col_name}: {e}")
        
        # Test specific collection name (you can change this)
        test_collection_name = input("\n6. Enter collection name to test (or press Enter to skip): ").strip()
        
        if test_collection_name:
            print(f"\n7. Testing collection '{test_collection_name}'...")
            
            # Find collection ID
            target_collection_id = None
            for col in collections:
                if col["name"] == test_collection_name:
                    target_collection_id = col["id"]
                    break
            
            if target_collection_id:
                print(f"✅ Found collection '{test_collection_name}' with ID: {target_collection_id}")
                
                # Filter documents for this collection
                filtered_docs = [doc for doc in vector_docs if doc.get("collectionId") == target_collection_id]
                print(f"✅ Found {len(filtered_docs)} documents in this collection")
                
                if filtered_docs:
                    print("   Documents in collection:")
                    for doc in filtered_docs[:10]:  # Show first 10
                        print(f"   - ID: {doc['id']}, Title: '{doc.get('title', 'No title')}', Type: {doc.get('source_type', 'unknown')}")
                else:
                    print("   ⚠️ No documents found in this collection!")
            else:
                print(f"❌ Collection '{test_collection_name}' not found!")
                print("   Available collections:")
                for col in collections:
                    print(f"   - '{col['name']}'")
        
        # Check for common issues
        print("\n8. Checking for common issues...")
        
        # Issue 1: No documents in vector store
        if not vector_docs:
            print("❌ ISSUE FOUND: No documents have in_vector_store = True")
            print("   Solution: Check if documents were properly processed and added to vector store")
        
        # Issue 2: Collection name mismatch
        if test_collection_name and not target_collection_id:
            print(f"❌ ISSUE FOUND: Collection name '{test_collection_name}' not found")
            print("   Solution: Use exact collection name from the list above")
        
        # Issue 3: Documents exist but not in target collection
        if test_collection_name and target_collection_id and not filtered_docs:
            print(f"❌ ISSUE FOUND: No documents in collection '{test_collection_name}'")
            print("   Solution: Check if documents were assigned to the correct collection")
        
        print("\n✅ Debug complete!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_metadata_loading()

