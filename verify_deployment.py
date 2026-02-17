
import requests
import sys

BASE_URL = "https://rag.aichieve.net"
# BASE_URL = "http://localhost:8080"

def check_endpoint(endpoint):
    url = f"{BASE_URL}{endpoint}"
    print(f"Checking {url}...", end=" ", flush=True)
    try:
        # Disable SSL verification for testing
        requests.packages.urllib3.disable_warnings()
        response = requests.get(url, timeout=10, verify=False)
        if response.status_code == 200:
            print(f"✅ {response.status_code}")
            try:
                print(f"   Response: {response.json()}")
            except:
                print(f"   Response: {response.text[:100]}...")
            return True
        else:
            print(f"❌ {response.status_code}")
            print(f"   Headers: {response.headers}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print(f"🚀 Verifying deployment at {BASE_URL}")
    print("=" * 50)
    # Context: Switched to internal port 80 to align with Coolify defaults
    
    endpoints = [
        "/",  # Root might show status or 404 depending on app
        "http://rag.aichieve.net/health", # Check if HTTP is working (ACME challenge path)
        "/health",  # Unconditional health check
        "/query_hybrid_enhanced/health",
        "/query_hybrid_enhanced/status", 
        "/query_hybrid_enhanced/ping",
        "/validate_knowledge_gap_setup"
    ]
    
    success_count = 0
    for endpoint in endpoints:
        if check_endpoint(endpoint):
            success_count += 1
            
    print("=" * 50)
    print(f"Summary: {success_count}/{len(endpoints)} endpoints accessible")

if __name__ == "__main__":
    main()
