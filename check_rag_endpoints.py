# check_rag_endpoints.py - Check what endpoints are available in your RAG server

import requests
import json

def check_rag_endpoints():
    """Check what endpoints are available in the running RAG server"""
    
    print("🔍 CHECKING RAG SERVER ENDPOINTS")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8080"
    
    # Common endpoints to test
    endpoints_to_test = [
        "/",
        "/health", 
        "/status",
        "/info",
        "/list_collections",
        "/collections",
        "/query",
        "/query_hybrid_enhanced",
        "/search",
        "/api/collections",
        "/api/query",
    ]
    
    print(f"🌐 Testing endpoints on {base_url}...")
    
    working_endpoints = []
    
    for endpoint in endpoints_to_test:
        try:
            url = f"{base_url}{endpoint}"
            print(f"   Testing {endpoint}...", end="")
            
            # Try GET first
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(" ✅ GET works")
                working_endpoints.append({
                    "endpoint": endpoint,
                    "method": "GET", 
                    "status": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "response_length": len(response.text)
                })
            elif response.status_code in [405]:  # Method not allowed, try POST
                try:
                    post_response = requests.post(url, json={}, timeout=5)
                    if post_response.status_code in [200, 400, 422]:  # 400/422 means it expects different data
                        print(f" ✅ POST works (status: {post_response.status_code})")
                        working_endpoints.append({
                            "endpoint": endpoint,
                            "method": "POST",
                            "status": post_response.status_code,
                            "content_type": post_response.headers.get("content-type", ""),
                            "response_length": len(post_response.text)
                        })
                    else:
                        print(f" ❌ {response.status_code}")
                except:
                    print(f" ❌ {response.status_code}")
            else:
                print(f" ❌ {response.status_code}")
                
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
    
    print(f"\\n📊 WORKING ENDPOINTS FOUND:")
    print("=" * 30)
    
    if working_endpoints:
        for ep in working_endpoints:
            print(f"✅ {ep['endpoint']} ({ep['method']}) - Status: {ep['status']}")
            if ep['response_length'] > 0:
                print(f"   Response length: {ep['response_length']} chars")
    else:
        print("❌ No working endpoints found")
    
    # Test a simple query to see what the actual endpoint might be
    print(f"\\n🧪 TESTING QUERY FUNCTIONALITY:")
    print("=" * 35)
    
    query_endpoints = ["/query_hybrid_enhanced", "/query", "/search", "/api/query"]
    
    for endpoint in query_endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"   Testing query on {endpoint}...")
            
            test_data = {
                "query": "test query",
                "collection_name": "main",
                "top_k": 1
            }
            
            response = requests.post(url, json=test_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Query successful! Response keys: {list(result.keys())}")
                return endpoint  # Return the working query endpoint
            elif response.status_code in [400, 422]:
                print(f"   ⚠️ Endpoint exists but data format issue (status: {response.status_code})")
                try:
                    error_detail = response.json()
                    print(f"      Error: {error_detail}")
                except:
                    print(f"      Raw response: {response.text[:200]}")
            else:
                print(f"   ❌ Status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    return None

def check_fastmcp_server_structure():
    """Check if this is a FastMCP server and what functions are available"""
    
    print(f"\\n🔧 CHECKING FastMCP SERVER STRUCTURE")
    print("=" * 40)
    
    try:
        # Try to access the FastMCP tools endpoint
        response = requests.get("http://127.0.0.1:8080/tools", timeout=5)
        
        if response.status_code == 200:
            tools = response.json()
            print("✅ FastMCP server detected!")
            print("Available tools:")
            for tool in tools:
                print(f"   📋 {tool.get('name', 'Unknown')} - {tool.get('description', 'No description')}")
            return tools
        else:
            print(f"❌ Not a FastMCP server (or different structure)")
            
    except Exception as e:
        print(f"❌ Error checking FastMCP structure: {str(e)}")
        
    return None

if __name__ == "__main__":
    working_query_endpoint = check_rag_endpoints()
    fastmcp_tools = check_fastmcp_server_structure()
    
    print(f"\\n🎯 SUMMARY:")
    if working_query_endpoint:
        print(f"✅ Found working query endpoint: {working_query_endpoint}")
    else:
        print("❌ No working query endpoint found")
        
    if fastmcp_tools:
        print(f"✅ FastMCP server with {len(fastmcp_tools)} tools available")
    else:
        print("❓ Server structure unknown")
        
    print(f"\\n💡 NEXT STEPS:")
    print("1. Use the working endpoints found above")
    print("2. Update the integration to use the correct endpoint structure")
    print("3. Check your RAG server's main.py or fastmcp_rag_server.py for the actual endpoint definitions")