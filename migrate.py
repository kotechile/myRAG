import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DB Connection String
# The .env file has DB_CONNECTION
DB_URL = os.getenv("DB_CONNECTION")
if not DB_URL:
    # Fallback or check for components
    print("❌ DB_CONNECTION not found in .env")
    # Try constructing from SUPABASE variables if needed, but DB_CONNECTION was seen in .env
    exit(1)

SQL_FILE = "restore_schema.sql"

def run_migration():
    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor()
        
        print(f"Reading {SQL_FILE}...")
        with open(SQL_FILE, 'r') as f:
            sql_content = f.read()
            
        print("Executing SQL script...")
        # Split by statements if needed, or execute block
        # psycopg2 can execute multiple statements in one go usually
        cur.execute(sql_content)
        
        print("✅ Migration executed successfully.")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        exit(1)

if __name__ == "__main__":
    run_migration()
