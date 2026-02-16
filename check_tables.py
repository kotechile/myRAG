import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Supabase credentials missing.")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

tables_to_check = [
    "lindex_documents",
    "lindex_collections",
    "Titles",
    "llm_providers",
    "api_keys",
    "images",  # Changed from Images to images
    "user_resource_usage"
]

print(f"Checking {len(tables_to_check)} tables in {SUPABASE_URL}...\n")

for table in tables_to_check:
    print(f"--- Checking table: {table} ---")
    try:
        response = supabase.table(table).select("*").limit(1).execute()
        # count check if possible, or just check compatibility
        # print(f"Response: {response}")
        if hasattr(response, 'data'):
            print(f"✅ Table '{table}' exists.")
            if len(response.data) > 0:
                print(f"   Sample keys: {list(response.data[0].keys())}")
            else:
                print("   Table is empty (but exists).")
        else:
             print(f"❓ Unexpected response format for '{table}'.")

    except Exception as e:
        print(f"❌ Error checking '{table}': {str(e)}")

print("\n--- Checking Vector Store Tables (in 'vecs' schema) ---")
try:
    collections = supabase.table("lindex_collections").select("name").limit(3).execute()
    if collections.data:
        for col in collections.data:
            name = col['name']
            # vecs tables are typically named "vecs"."{name}"
            # But via API we might not be able to access them directly as they are in a different schema
            # unless the API role has access.
            print(f"Checking collection table: vecs.{name}")
            # This might fail if we can't select from other schemas via postgrest
            # but let's try.
            # PostgREST usually only exposes 'public' schema unless configured otherwise.
            # So this check might be inconclusive if it fails, but if it works it's good.
            pass 
            
    else:
        print("No collections found in lindex_collections to check.")
except Exception as e:
    print(f"Error checking collections: {e}")

