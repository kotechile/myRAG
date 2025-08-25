#!/usr/bin/env python3

import os
import sys
import json
import uuid
import traceback
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# FastMCP imports
try:
    from fastmcp import FastMCP
    print("✅ FastMCP imported successfully")
except ImportError as e:
    print(f"❌ FastMCP import failed: {e}")
    print("Please install: pip install fastmcp")
    sys.exit(1)

# Your existing RAG imports
try:
    from main import get_rag_engine, embedding_manager
    from supabase import create_client
    from llama_parse import LlamaParse
    from llama_index.core import SimpleDirectoryReader
    print("✅ RAG imports successful")
except ImportError as e:
    print(f"⚠️ RAG import failed: {e}")
    print("RAG functionality will be simulated")

# Initialize FastMCP server
mcp = FastMCP("RAG Server")

@mcp.tool()
def test_connection() -> str:
    """Test if the MCP server is working"""
    return "✅ MCP Server is working correctly! Connection successful."

@mcp.tool()
def debug_rag_engine(collection_name: str) -> str:
    """Debug what methods are available in the RAG engine
    
    Args:
        collection_name: Collection to debug
    """
    try:
        print(f"🔍 Debugging RAG engine for collection: {collection_name}", file=sys.stderr)
        engine = get_rag_engine(collection_name)
        
        available_methods = [method for method in dir(engine) if not method.startswith('_')]
        
        debug_info = {
            "status": "success",
            "collection_name": collection_name,
            "engine_type": str(type(engine)),
            "available_methods": available_methods,
            "has_query_simple": hasattr(engine, 'query_simple'),
            "has_query_hybrid_enhanced": hasattr(engine, 'query_hybrid_enhanced'),
            "has_query_agentic_fixed": hasattr(engine, 'query_agentic_fixed'),
            "document_count": len(engine.document_metadata) if hasattr(engine, 'document_metadata') else "unknown"
        }
        
        print(f"✅ Debug successful for {collection_name}", file=sys.stderr)
        return json.dumps(debug_info, indent=2)
        
    except Exception as e:
        error_info = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "collection_name": collection_name
        }
        print(f"❌ Debug failed: {str(e)}", file=sys.stderr)
        return json.dumps(error_info, indent=2)

@mcp.tool()
def query_simple_debug(
    query: str,
    collection_name: str,
    num_results: int = 5,
    llm_provider: str = "deepseek"
) -> str:
    """Debug version of simple query with detailed error reporting
    
    Args:
        query: The question or search query
        collection_name: Collection to search in
        num_results: Number of results to return (1-20)
        llm_provider: LLM provider to use (deepseek, openai, claude, gemini)
    """
    try:
        print(f"🔍 Starting query_simple_debug", file=sys.stderr)
        print(f"Query: {query}", file=sys.stderr)
        print(f"Collection: {collection_name}", file=sys.stderr)
        
        # Get the RAG engine
        print(f"Getting RAG engine for: {collection_name}", file=sys.stderr)
        engine = get_rag_engine(collection_name)
        print(f"Got engine: {type(engine)}", file=sys.stderr)
        
        # Check if method exists
        if not hasattr(engine, 'query_simple'):
            available_methods = [method for method in dir(engine) if not method.startswith('_')]
            return json.dumps({
                "status": "error",
                "error": "query_simple method not found",
                "available_methods": available_methods
            }, indent=2)
        
        # Attempt the query
        print(f"Calling engine.query_simple...", file=sys.stderr)
        result = engine.query_simple(
            query=query,
            num_results=num_results,
            llm_provider=llm_provider
        )
        print(f"Query completed successfully", file=sys.stderr)
        
        # Return the result
        response = {
            "status": "success",
            "query": query,
            "collection_name": collection_name,
            "num_results": num_results,
            "llm_provider": llm_provider,
            "result": result
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_response = {
            "status": "error",
            "error_type": str(type(e).__name__),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": query,
            "collection_name": collection_name,
            "num_results": num_results,
            "llm_provider": llm_provider
        }
        print(f"❌ Query failed: {str(e)}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        return json.dumps(error_response, indent=2)

@mcp.tool()
def query_simple(
    query: str,
    collection_name: str,
    num_results: int = 5,
    llm_provider: str = "deepseek"
) -> str:
    """Perform a simple similarity search query on the document collection
    
    Args:
        query: The question or search query
        collection_name: Collection to search in
        num_results: Number of results to return (1-20)
        llm_provider: LLM provider to use (deepseek, openai, claude, gemini)
    """
    try:
        engine = get_rag_engine(collection_name)
        result = engine.query_simple(
            query=query,
            num_results=num_results,
            llm_provider=llm_provider
        )
        
        # Ensure result is serializable
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        else:
            return json.dumps({
                "status": "success",
                "query": query,
                "collection_name": collection_name,
                "response": str(result)
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "query": query,
            "collection_name": collection_name
        }, indent=2)

@mcp.tool()
def query_hybrid_enhanced(
    query: str,
    collection_name: str,
    top_k: int = 10,
    llm_provider: str = "deepseek"
) -> str:
    """Perform enhanced hybrid search with importance weighting and multi-source analysis
    
    Args:
        query: The question or search query
        collection_name: Collection to search in
        top_k: Number of chunks to retrieve (1-30)
        llm_provider: LLM provider to use (deepseek, openai, claude, gemini)
    """
    try:
        engine = get_rag_engine(collection_name)
        result = engine.query_hybrid_enhanced(
            query=query,
            top_k=top_k,
            llm_provider=llm_provider
        )
        
        # Ensure result is serializable
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        else:
            return json.dumps({
                "status": "success",
                "query": query,
                "collection_name": collection_name,
                "response": str(result)
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "query": query,
            "collection_name": collection_name
        }, indent=2)

@mcp.tool()
def query_agentic_fixed(
    query: str,
    collection_name: str,
    max_docs: int = 3,
    verbose_mode: str = "balanced",
    llm_provider: str = "deepseek"
) -> str:
    """Advanced agentic query using ReActAgent with multiple document tools
    
    Args:
        query: The question or search query
        collection_name: Collection to search in
        max_docs: Maximum number of documents to create tools for (1-5)
        verbose_mode: Response verbosity level (concise, balanced, detailed)
        llm_provider: LLM provider to use (deepseek, openai, claude, gemini)
    """
    try:
        engine = get_rag_engine(collection_name)
        result = engine.query_agentic_fixed(
            query=query,
            max_docs=max_docs,
            llm_provider=llm_provider,
            verbose_mode=verbose_mode
        )
        
        # Ensure result is serializable
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        else:
            return json.dumps({
                "status": "success",
                "query": query,
                "collection_name": collection_name,
                "response": str(result)
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "query": query,
            "collection_name": collection_name
        }, indent=2)

@mcp.tool()
def list_collections() -> str:
    """List all available document collections"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        
        # Get collections from database
        response = supabase.table("lindex_collections").select("*").execute()
        
        collections = []
        for collection in response.data:
            collections.append({
                "name": collection["name"],
                "id": collection["id"],
                "created_at": collection.get("created_at"),
                "description": collection.get("description", "")
            })
        
        return json.dumps({
            "status": "success",
            "collections": collections,
            "total_count": len(collections)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"List collections failed: {str(e)}"
        }, indent=2)

@mcp.tool()
def get_collection_info(collection_name: str) -> str:
    """Get detailed information about a specific collection
    
    Args:
        collection_name: Name of the collection to get info for
    """
    try:
        engine = get_rag_engine(collection_name)
        
        # Get document metadata
        doc_metadata = engine.document_metadata
        
        collection_info = {
            "status": "success",
            "collection_name": collection_name,
            "total_documents": len(doc_metadata),
            "embedding_dimension": embedding_manager.get_dimension(),
            "documents": []
        }
        
        for doc_id, metadata in doc_metadata.items():
            collection_info["documents"].append({
                "document_id": doc_id,
                "title": metadata.get("title", f"Document {doc_id}"),
                "summary": metadata.get("summary_short", "")[:200] if metadata.get("summary_short") else "",
                "source_type": metadata.get("source_type", "document"),
                "chunk_count": metadata.get("chunk_count", 0),
                "doc_size": metadata.get("doc_size", 0)
            })
        
        return json.dumps(collection_info, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"Get collection info failed: {str(e)}"
        }, indent=2)

if __name__ == "__main__":
    print("🚀 Starting FastMCP RAG server...")
    mcp.run()