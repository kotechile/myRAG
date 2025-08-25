# embedding_verification.py - Verify if optimization is working

import os
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
import vecs
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment
load_dotenv()

# Database connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DATABASE_PASSWORD = os.getenv("SUPABASE_DATABASE_PASSWORD")
DB_CONNECTION = f"postgresql://postgres.dgcsqiaciyqvprtpopxg:{SUPABASE_DATABASE_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:5432/postgres"

def check_embedding_dimensions_in_database(collection_name: str):
    """Check what embedding dimensions are actually stored in the database."""
    print(f"\n🔍 Checking embeddings in collection: {collection_name}")
    
    try:
        # Connect to vecs
        vx = vecs.Client(DB_CONNECTION)
        
        # Try different possible dimensions
        possible_dimensions = [128, 384, 512, 768, 1024, 1536]
        
        for dim in possible_dimensions:
            try:
                collection = vx.get_collection(name=collection_name)
                if collection:
                    print(f"✅ Found collection '{collection_name}'")
                    
                    # Get some sample records
                    records = collection.query(
                        data=None,
                        limit=3,
                        include_value=True,
                        include_metadata=True
                    )
                    
                    if records:
                        print(f"📊 Sample records found: {len(records)}")
                        
                        for i, record in enumerate(records):
                            # Check embedding dimension
                            if hasattr(record, 'vec') and record.vec is not None:
                                embedding_dim = len(record.vec)
                                print(f"  Record {i+1}: Embedding dimension = {embedding_dim}")
                                
                                # Check if values look quantized (int8)
                                embedding_array = np.array(record.vec)
                                unique_values = len(np.unique(embedding_array))
                                value_range = (embedding_array.min(), embedding_array.max())
                                
                                print(f"    Unique values: {unique_values}")
                                print(f"    Value range: {value_range[0]:.6f} to {value_range[1]:.6f}")
                                
                                # Check if values are quantized (limited unique values)
                                if unique_values < 500:  # Quantized embeddings have limited unique values
                                    print(f"    🎯 LIKELY QUANTIZED (only {unique_values} unique values)")
                                else:
                                    print(f"    📊 Standard float embeddings ({unique_values} unique values)")
                                
                                # Check metadata for optimization info
                                if hasattr(record, 'metadata') and record.metadata:
                                    metadata = record.metadata
                                    print(f"    Metadata keys: {list(metadata.keys())}")
                                    
                                    # Look for optimization indicators
                                    optimization_keys = ['optimization_type', 'original_dim', 'compressed_dim', 'quantization']
                                    for key in optimization_keys:
                                        if key in metadata:
                                            print(f"    🔧 {key}: {metadata[key]}")
                                
                                print()
                    break
                    
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"❌ Error checking database: {str(e)}")


def test_embedding_models_directly():
    """Test embedding models directly to see their output."""
    print("\n🧪 Testing embedding models directly...")
    
    test_text = "This is a test sentence for embedding comparison."
    
    # Test standard OpenAI embedding
    print("\n1️⃣ Standard OpenAI Embedding:")
    standard_model = OpenAIEmbedding(model="text-embedding-3-small")
    standard_embedding = standard_model.get_text_embedding(test_text)
    
    print(f"  Dimension: {len(standard_embedding)}")
    print(f"  Sample values: {standard_embedding[:5]}")
    print(f"  Value range: {min(standard_embedding):.6f} to {max(standard_embedding):.6f}")
    print(f"  Unique values: {len(np.unique(np.array(standard_embedding)))}")
    
    # Test if optimization is available
    try:
        from embedding_optimizer import OptimizedOpenAIEmbedding, CombinedOptimizer, create_128d_int8_optimizer
        
        print("\n2️⃣ Optimized Embedding (if available):")
        
        # Create optimizer
        optimizer = create_128d_int8_optimizer()
        
        # Fit optimizer with some sample data
        sample_embeddings = []
        sample_texts = [
            "Sample text one for training",
            "Another sample text for fitting",
            "Third sample for the optimizer",
            "Fourth text sample",
            "Fifth sample text"
        ]
        
        for text in sample_texts:
            emb = standard_model.get_text_embedding(text)
            sample_embeddings.append(emb)
        
        # Fit the optimizer
        sample_array = np.array(sample_embeddings)
        optimizer.fit(sample_array)
        
        # Test optimized embedding
        optimized_model = OptimizedOpenAIEmbedding(optimizer=optimizer)
        optimized_embedding = optimized_model.get_text_embedding(test_text)
        
        print(f"  Dimension: {len(optimized_embedding)}")
        print(f"  Sample values: {optimized_embedding[:5]}")
        print(f"  Value range: {min(optimized_embedding):.6f} to {max(optimized_embedding):.6f}")
        print(f"  Unique values: {len(np.unique(np.array(optimized_embedding)))}")
        
        # Compare
        print(f"\n📊 Comparison:")
        print(f"  Standard dim: {len(standard_embedding)} → Optimized dim: {len(optimized_embedding)}")
        print(f"  Compression ratio: {len(standard_embedding) / len(optimized_embedding):.1f}x")
        
        # Check if actually different
        if len(standard_embedding) != len(optimized_embedding):
            print(f"  ✅ Optimization is WORKING (different dimensions)")
        else:
            print(f"  ❌ Optimization might NOT be working (same dimensions)")
            
    except ImportError:
        print("\n❌ Optimization module not available")
    except Exception as e:
        print(f"\n❌ Error testing optimization: {str(e)}")


def check_collection_schema(collection_name: str):
    """Check the database schema for the collection."""
    print(f"\n🏗️ Checking database schema for collection: {collection_name}")
    
    try:
        import psycopg2
        
        # Connect directly to PostgreSQL
        conn = psycopg2.connect(DB_CONNECTION)
        cursor = conn.cursor()
        
        # Check if table exists and get its schema
        table_name = f"vecs_{collection_name}"
        
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        
        columns = cursor.fetchall()
        
        if columns:
            print(f"✅ Found table: {table_name}")
            print("📋 Schema:")
            for col_name, data_type, max_length in columns:
                if max_length:
                    print(f"  {col_name}: {data_type}({max_length})")
                else:
                    print(f"  {col_name}: {data_type}")
                    
            # Check vector dimension specifically
            cursor.execute(f"""
                SELECT pg_get_expr(atttypmod, attrelid) as typmod
                FROM pg_attribute 
                WHERE attrelid = '{table_name}'::regclass 
                AND attname = 'vec';
            """)
            
            vec_info = cursor.fetchone()
            if vec_info and vec_info[0]:
                print(f"🎯 Vector dimension from schema: {vec_info[0]}")
        else:
            print(f"❌ Table {table_name} not found")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking schema: {str(e)}")


def main():
    """Main verification function."""
    print("🔍 EMBEDDING OPTIMIZATION VERIFICATION")
    print("=" * 50)
    
    # Get collection name from user
    collection_name = input("Enter your collection name (e.g., 'default_collection'): ").strip()
    
    if not collection_name:
        collection_name = "default_collection"
        print(f"Using default collection: {collection_name}")
    
    # Run all checks
    test_embedding_models_directly()
    check_collection_schema(collection_name)
    check_embedding_dimensions_in_database(collection_name)
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY:")
    print("- If you see dimension 1536: Standard OpenAI embeddings")
    print("- If you see dimension 128 or 384: Likely optimized embeddings")
    print("- If you see <500 unique values: Likely quantized embeddings")
    print("- If you see >10000 unique values: Standard float embeddings")


if __name__ == "__main__":
    main()