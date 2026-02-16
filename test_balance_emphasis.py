#!/usr/bin/env python3
"""
Test script to demonstrate the new balance emphasis functionality.
This script shows how different balance settings affect document prioritization.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
COLLECTION_NAME = "your_collection_name"  # Replace with your actual collection name

def test_balance_emphasis():
    """Test different balance emphasis settings."""
    
    # Test queries that should trigger different balance settings
    test_queries = [
        {
            "query": "What are the latest developments in AI technology in 2024?",
            "expected_balance": "news_focused",
            "description": "News-focused query - should favor recent articles"
        },
        {
            "query": "Give me a comprehensive analysis of machine learning algorithms",
            "expected_balance": "comprehensive", 
            "description": "Comprehensive query - should favor books and detailed sources"
        },
        {
            "query": "How does neural network training work?",
            "expected_balance": "balanced",
            "description": "General query - should use balanced approach"
        }
    ]
    
    print("🧪 Testing Balance Emphasis Functionality")
    print("=" * 50)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected balance: {test_case['expected_balance']}")
        
        # Test with auto-detection (no balance_emphasis parameter)
        print("\n🔍 Testing with auto-detection...")
        result_auto = make_query(test_case['query'], method='simple')
        
        # Test with explicit balance settings
        for balance in ['news_focused', 'balanced', 'comprehensive']:
            print(f"\n🔍 Testing with explicit '{balance}' setting...")
            result_explicit = make_query(test_case['query'], method='simple', balance_emphasis=balance)
            
            # Compare results
            if result_auto and result_explicit:
                print(f"  Auto-detected: {len(result_auto.get('documents_used', []))} documents")
                print(f"  {balance}: {len(result_explicit.get('documents_used', []))} documents")
                
                # Show document types used
                auto_docs = result_auto.get('documents_used', [])
                explicit_docs = result_explicit.get('documents_used', [])
                
                print(f"  Auto-detected doc types: {[doc.get('source_type', 'unknown') for doc in auto_docs]}")
                print(f"  {balance} doc types: {[doc.get('source_type', 'unknown') for doc in explicit_docs]}")
        
        print("-" * 30)

def make_query(query, method='simple', balance_emphasis=None, num_results=3):
    """Make a query to the RAG system."""
    try:
        url = f"{BASE_URL}/query_{method}"
        payload = {
            "query": query,
            "collection_name": COLLECTION_NAME,
            "num_results": num_results
        }
        
        if balance_emphasis:
            payload["balance_emphasis"] = balance_emphasis
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test_balance_comparison():
    """Compare results with different balance settings for the same query."""
    
    query = "What are the latest trends in artificial intelligence?"
    
    print(f"\n🔄 Balance Comparison Test")
    print(f"Query: {query}")
    print("=" * 50)
    
    balance_settings = ['news_focused', 'balanced', 'comprehensive']
    
    for balance in balance_settings:
        print(f"\n📊 Testing with '{balance}' emphasis:")
        result = make_query(query, method='hybrid_enhanced', balance_emphasis=balance, num_results=5)
        
        if result and result.get('status') == 'success':
            documents = result.get('documents_used', [])
            print(f"  Found {len(documents)} documents")
            
            # Analyze document types
            doc_types = {}
            for doc in documents:
                doc_type = doc.get('source_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            print(f"  Document types: {doc_types}")
            
            # Show importance weights
            weights = [doc.get('importance_weight', 1.0) for doc in documents]
            print(f"  Importance weights: {[round(w, 2) for w in weights]}")
            
        else:
            print(f"  Query failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("🚀 Starting Balance Emphasis Tests")
    print("Make sure your RAG server is running on http://localhost:5000")
    print("Update COLLECTION_NAME variable with your actual collection name")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            exit(1)
    except:
        print("❌ Cannot connect to server. Make sure it's running on http://localhost:5000")
        exit(1)
    
    # Run tests
    test_balance_emphasis()
    test_balance_comparison()
    
    print("\n✅ Tests completed!")
    print("\n💡 Usage Tips:")
    print("1. Use 'news_focused' for current events and recent developments")
    print("2. Use 'comprehensive' for detailed analysis and research")
    print("3. Use 'balanced' for general queries (or let auto-detection handle it)")
    print("4. The system will auto-detect the appropriate balance if not specified")

