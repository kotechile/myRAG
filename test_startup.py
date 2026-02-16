#!/usr/bin/env python3
"""
Test script to verify the system startup fixes
"""

import sys
import os
import time
import signal
import subprocess
from pathlib import Path

def test_startup():
    """Test the system startup with timeout"""
    print("🧪 Testing system startup...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)
    
    try:
        # Start the process with timeout
        process = subprocess.Popen(
            [sys.executable, 'main.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup with timeout
        print("⏳ Waiting for startup (30 seconds timeout)...")
        try:
            stdout, stderr = process.communicate(timeout=30)
            print("✅ Process completed normally")
            print(f"Exit code: {process.returncode}")
            if stdout:
                print("STDOUT:", stdout[-500:])  # Last 500 chars
            if stderr:
                print("STDERR:", stderr[-500:])  # Last 500 chars
        except subprocess.TimeoutExpired:
            print("⏰ Process timed out - this might indicate a hang")
            process.kill()
            stdout, stderr = process.communicate()
            print("STDOUT:", stdout[-500:] if stdout else "None")
            print("STDERR:", stderr[-500:] if stderr else "None")
            return False
            
    except Exception as e:
        print(f"❌ Error starting process: {e}")
        return False
    
    return process.returncode == 0

def test_health_endpoints():
    """Test health check endpoints"""
    print("\n🏥 Testing health check endpoints...")
    
    import requests
    import time
    
    base_url = "http://localhost:8080"
    endpoints = [
        "/query_hybrid_enhanced/health",
        "/query_hybrid_enhanced/status", 
        "/query_hybrid_enhanced/ping",
        "/query_hybrid_enhanced/",
        "/validate_knowledge_gap_setup"
    ]
    
    # Wait a bit for server to start
    time.sleep(2)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint}: {e}")

if __name__ == "__main__":
    print("🚀 RAG System Startup Test")
    print("=" * 50)
    
    # Test 1: Startup
    success = test_startup()
    
    if success:
        print("\n✅ Startup test passed!")
    else:
        print("\n❌ Startup test failed!")
        sys.exit(1)
    
    print("\n🎉 All tests completed!")





