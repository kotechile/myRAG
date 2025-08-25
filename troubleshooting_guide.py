#!/usr/bin/env python3
"""
Complete troubleshooting guide for Knowledge Gap Closure System
Run with: python troubleshooting_guide.py [command]
"""

import sys
import os
import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# CRITICAL FIX: Load environment variables from .env file
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not available. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

class Colors:
    """ANSI color codes for pretty output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title:^60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_step(step: str, status: str = "info"):
    """Print a step with status indicator"""
    if status == "success":
        icon = f"{Colors.GREEN}✅{Colors.END}"
    elif status == "error":
        icon = f"{Colors.RED}❌{Colors.END}"
    elif status == "warning":
        icon = f"{Colors.YELLOW}⚠️{Colors.END}"
    else:
        icon = f"{Colors.BLUE}🔍{Colors.END}"
    
    print(f"{icon} {step}")

def check_environment_variables():
    """Check if required environment variables are set"""
    print_step("Checking environment variables...")
    
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'SUPABASE_DATABASE_PASSWORD',
        'OPENAI_API_KEY',
        'DEEPSEEK_API_KEY'
    ]
    
    optional_vars = [
        'LLAMA_CLOUD_API_KEY',
        'LINKUP_API_KEY'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
        else:
            print_step(f"   {var}: ✓ Set", "success")
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
        else:
            print_step(f"   {var}: ✓ Set", "success")
    
    if missing_required:
        print_step(f"Missing required variables: {missing_required}", "error")
        print(f"{Colors.YELLOW}Add these to your .env file:{Colors.END}")
        for var in missing_required:
            print(f"   {var}=your_value_here")
        return False
    
    if missing_optional:
        print_step(f"Missing optional variables: {missing_optional}", "warning")
        print(f"{Colors.YELLOW}These are optional but recommended for full functionality{Colors.END}")
    
    print_step("Environment variables check complete", "success")
    return True

def check_flask_server(base_url: str = "http://localhost:8080"):
    """Check if Flask server is running and accessible"""
    print_step("Checking Flask server connectivity...")
    
    try:
        response = requests.get(f"{base_url}/embedding_status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_step("Server is running and accessible", "success")
            print(f"   - Optimization available: {data.get('optimization_available', 'unknown')}")
            print(f"   - Current dimension: {data.get('current_dimension', 'unknown')}")
            print(f"   - Agents available: {data.get('agents_available', 'unknown')}")
            return True
        else:
            print_step(f"Server responded with status {response.status_code}", "error")
            return False
    except requests.exceptions.ConnectionError:
        print_step("Cannot connect to server", "error")
        print(f"{Colors.YELLOW}   Start the server with: python main.py{Colors.END}")
        return False
    except Exception as e:
        print_step(f"Error testing server: {e}", "error")
        return False

def test_gap_closure_endpoints(base_url: str = "http://localhost:8080"):
    """Test all gap closure endpoints"""
    print_step("Testing gap closure endpoints...")
    
    endpoints = [
        ("Gap Closure Status", "GET", "/gap_closure_status"),
        ("Open Gaps Analysis", "GET", "/analyze_knowledge_gaps_open_only"),
        ("Gap Filler Status", "GET", "/gap_filler_status")
    ]
    
    results = {}
    all_working = True
    
    for name, method, endpoint in endpoints:
        try:
            print_step(f"   Testing {name}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print_step(f"   {name}: Working", "success")
                
                # Show relevant info
                if name == "Gap Closure Status":
                    if "gap_closure_status" in data:
                        status_data = data["gap_closure_status"]
                        if isinstance(status_data, dict):
                            print(f"      - Total titles: {status_data.get('total_titles', 'unknown')}")
                            print(f"      - Gaps closed: {status_data.get('gaps_closed_count', 'unknown')}")
                            print(f"      - Gaps open: {status_data.get('gaps_open_count', 'unknown')}")
                
                elif name == "Open Gaps Analysis":
                    gaps_found = data.get('gaps_found', 0)
                    print(f"      - Open gaps found: {gaps_found}")
                
                elif name == "Gap Filler Status":
                    if "document_statistics" in data:
                        stats = data["document_statistics"]
                        print(f"      - Gap filler docs: {stats.get('total', 0)}")
                        print(f"      - In vector store: {stats.get('in_vector_store', 0)}")
                
                results[name] = {"status": "success", "data": data}
            else:
                print_step(f"   {name}: HTTP {response.status_code}", "error")
                try:
                    error_data = response.json()
                    print(f"      Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"      Raw response: {response.text[:100]}")
                results[name] = {"status": "error", "code": response.status_code}
                all_working = False
                
        except Exception as e:
            print_step(f"   {name}: Connection error - {str(e)}", "error")
            results[name] = {"status": "error", "error": str(e)}
            all_working = False
    
    if all_working:
        print_step("All gap closure endpoints working!", "success")
    else:
        print_step("Some endpoints have issues", "warning")
    
    return results

def test_rag_endpoints(base_url: str = "http://localhost:8080"):
    """Test basic RAG endpoints"""
    print_step("Testing basic RAG endpoints...")
    
    # Test a simple query
    test_data = {
        "query": "What are the key things to consider before buying a house?",
        "collection_name": "rag_house_and_real_estate",
        "llm": "deepseek",
        "num_results": 3
    }
    
    try:
        response = requests.post(f"{base_url}/query_simple", json=test_data, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print_step("RAG query system working", "success")
                print(f"   - Response length: {len(data.get('response', ''))}")
                print(f"   - Chunks used: {data.get('chunks_used', 0)}")
                print(f"   - Documents searched: {data.get('documents_searched', 0)}")
                return True
            else:
                print_step(f"RAG query failed: {data.get('error', 'Unknown error')}", "error")
                return False
        else:
            print_step(f"RAG endpoint error: HTTP {response.status_code}", "error")
            return False
            
    except Exception as e:
        print_step(f"RAG test error: {e}", "error")
        return False

def run_comprehensive_test():
    """Run comprehensive system test"""
    print_step("Running comprehensive system test...", "info")
    
    # Test database schema
    print_step("Testing database schema (requires running server)...")
    if check_flask_server():
        gap_results = test_gap_closure_endpoints()
        rag_working = test_rag_endpoints()
        
        print_step("=== TEST SUMMARY ===", "info")
        
        if all(result.get("status") == "success" for result in gap_results.values()):
            print_step("✅ All gap closure endpoints working", "success")
        else:
            print_step("❌ Some gap closure endpoints failing", "error")
        
        if rag_working:
            print_step("✅ RAG system working", "success")
        else:
            print_step("❌ RAG system has issues", "error")
        
        return True
    else:
        print_step("Cannot test endpoints - server not running", "error")
        return False

def show_usage():
    """Show usage information"""
    print_header("Knowledge Gap Closure System - Troubleshooting Guide")
    
    print(f"{Colors.BOLD}Usage:{Colors.END}")
    print(f"  python troubleshooting_guide.py [command]")
    print()
    
    print(f"{Colors.BOLD}Commands:{Colors.END}")
    print(f"  {Colors.GREEN}diagnose{Colors.END}     - Run comprehensive system diagnosis")
    print(f"  {Colors.GREEN}env{Colors.END}         - Check environment variables")
    print(f"  {Colors.GREEN}server{Colors.END}      - Check if server is running")
    print(f"  {Colors.GREEN}gaps{Colors.END}        - Test gap closure endpoints")
    print(f"  {Colors.GREEN}rag{Colors.END}         - Test RAG endpoints")
    print(f"  {Colors.GREEN}fix{Colors.END}         - Show common fixes")
    print()
    
    print(f"{Colors.BOLD}Examples:{Colors.END}")
    print(f"  python troubleshooting_guide.py diagnose")
    print(f"  python troubleshooting_guide.py env")
    print(f"  python troubleshooting_guide.py server")

def show_common_fixes():
    """Show common fixes for issues"""
    print_header("Common Fixes")
    
    fixes = [
        {
            "issue": "Server won't start",
            "solutions": [
                "Check environment variables are set in .env file",
                "Ensure port 8080 is not in use: lsof -i :8080",
                "Check for Python import errors: python -c 'import main'",
                "Verify dependencies: pip install -r requirements.txt"
            ]
        },
        {
            "issue": "Gap closure endpoints return 404",
            "solutions": [
                "Ensure knowledge_gap_http_supabase.py is in your directory",
                "Check that KNOWLEDGE_GAP_AVAILABLE is True in server logs",
                "Verify no duplicate route errors in server startup logs",
                "Restart the server after fixing imports"
            ]
        },
        {
            "issue": "Database connection errors",
            "solutions": [
                "Verify SUPABASE_URL and SUPABASE_KEY are correct",
                "Check SUPABASE_DATABASE_PASSWORD is set",
                "Test database connection independently",
                "Ensure Supabase project is active"
            ]
        },
        {
            "issue": "RAG queries fail",
            "solutions": [
                "Check if collection exists: verify collection_name",
                "Ensure documents are processed and in vector store",
                "Verify embedding dimension matches",
                "Check LLM provider API keys (OPENAI_API_KEY, DEEPSEEK_API_KEY)"
            ]
        }
    ]
    
    for fix in fixes:
        print(f"{Colors.BOLD}{Colors.RED}Issue: {fix['issue']}{Colors.END}")
        for i, solution in enumerate(fix['solutions'], 1):
            print(f"  {i}. {solution}")
        print()

def run_diagnose():
    """Run complete diagnosis"""
    print_header("System Diagnosis")
    
    print_step("🔍 Starting comprehensive diagnosis...", "info")
    
    # Step 1: Environment
    env_ok = check_environment_variables()
    
    # Step 2: Server
    server_ok = False
    if env_ok:
        server_ok = check_flask_server()
    
    # Step 3: Endpoints
    endpoints_ok = False
    rag_ok = False
    if server_ok:
        gap_results = test_gap_closure_endpoints()
        endpoints_ok = all(result.get("status") == "success" for result in gap_results.values())
        rag_ok = test_rag_endpoints()
    
    # Final summary
    print_header("Diagnosis Summary")
    
    status_items = [
        ("Environment Variables", env_ok),
        ("Flask Server", server_ok),
        ("Gap Closure Endpoints", endpoints_ok),
        ("RAG System", rag_ok)
    ]
    
    for item, status in status_items:
        if status:
            print_step(f"{item}: Working", "success")
        else:
            print_step(f"{item}: Issues detected", "error")
    
    if all(status for _, status in status_items):
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All systems working correctly!{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ Some issues detected. Run 'python troubleshooting_guide.py fix' for solutions.{Colors.END}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "diagnose":
        run_diagnose()
    elif command == "env":
        print_header("Environment Check")
        check_environment_variables()
    elif command == "server":
        print_header("Server Check")
        check_flask_server()
    elif command == "gaps":
        print_header("Gap Closure Endpoints Test")
        test_gap_closure_endpoints()
    elif command == "rag":
        print_header("RAG System Test")
        test_rag_endpoints()
    elif command == "fix":
        show_common_fixes()
    else:
        print(f"{Colors.RED}Unknown command: {command}{Colors.END}")
        show_usage()

if __name__ == "__main__":
    main()