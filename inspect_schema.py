import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DB Connection String
DB_URL = os.getenv("DB_CONNECTION")

def inspect_schema():
    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # 1. Inspect 'images' columns
        print("\n--- Table: images ---")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = 'images';
        """)
        columns = cur.fetchall()
        if columns:
            for col in columns:
                print(f"  {col[0]}: {col[1]}")
        else:
            print("  Table not found or no columns.")

        # 2. Inspect 'user_resource_usage' columns
        print("\n--- Table: user_resource_usage ---")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = 'user_resource_usage';
        """)
        columns = cur.fetchall()
        if columns:
            for col in columns:
                print(f"  {col[0]}: {col[1]}")
        else:
            print("  Table not found or no columns.")
            
        # 3. Check for 'match_documents' function
        print("\n--- Function: match_documents ---")
        cur.execute("""
            SELECT routine_name, routine_definition
            FROM information_schema.routines
            WHERE routine_schema = 'public' AND routine_name = 'match_documents';
        """)
        funcs = cur.fetchall()
        if funcs:
            print("  ✅ Function 'match_documents' exists.")
        else:
            print("  ❌ Function 'match_documents' DOES NOT exist.")

        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")

if __name__ == "__main__":
    inspect_schema()
