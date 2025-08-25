# create_optimizer.py - Create and save a fitted optimizer

import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def create_and_save_optimizer():
    """Create, fit, and save an optimizer."""
    
    print("🔧 CREATING FITTED OPTIMIZER")
    print("=" * 40)
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment")
        print("💡 Set your OpenAI API key first")
        return False
    
    try:
        from embedding_optimizer import fit_and_save_optimizer
        
        print("📊 This will:")
        print("  - Generate ~300 sample texts")
        print("  - Create embeddings using OpenAI API (~$1-2 cost)")
        print("  - Fit PCA (1536D → 128D) + int8 quantization")
        print("  - Save optimizer to optimizer_128d_int8.joblib")
        
        # Confirm with user
        confirm = input("\n🤔 Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled by user")
            return False
        
        print("\n🚀 Creating optimizer...")
        print("⏳ This may take 2-3 minutes due to API calls...")
        
        # Create the optimizer
        optimizer = fit_and_save_optimizer(
            sample_texts=None,  # Will auto-generate
            target_dim=128,
            quantization="int8",
            save_path="optimizer_128d_int8.joblib"
        )
        
        print("\n✅ SUCCESS!")
        print(f"📁 Optimizer saved to: optimizer_128d_int8.joblib")
        print(f"🎯 Compression: 1536D → 128D (12x smaller)")
        print(f"🔢 Quantization: float32 → int8 (4x smaller)")
        print(f"🚀 Total compression: ~48x smaller embeddings!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating optimizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_saved_optimizer():
    """Test the saved optimizer."""
    
    print("\n🧪 TESTING SAVED OPTIMIZER")
    print("=" * 40)
    
    try:
        from embedding_optimizer import CombinedOptimizer, OptimizedOpenAIEmbedding
        
        # Load the saved optimizer
        if not os.path.exists("optimizer_128d_int8.joblib"):
            print("❌ No saved optimizer found")
            return False
        
        optimizer = CombinedOptimizer.load("optimizer_128d_int8.joblib")
        print(f"✅ Loaded optimizer: {optimizer.target_dim}D + {optimizer.quantization}")
        
        # Test with a sample text
        optimized_model = OptimizedOpenAIEmbedding(optimizer=optimizer)
        test_embedding = optimized_model.get_text_embedding("This is a test text.")
        
        print(f"🎯 Test embedding dimension: {len(test_embedding)}")
        print(f"📊 Sample values: {test_embedding[:5]}")
        
        # Check if quantized
        import numpy as np
        unique_values = len(np.unique(test_embedding))
        print(f"🔢 Unique values: {unique_values}")
        
        if len(test_embedding) == 128 and unique_values < 500:
            print("✅ Optimization is working correctly!")
            return True
        else:
            print("⚠️ Optimization might not be working as expected")
            return False
            
    except Exception as e:
        print(f"❌ Error testing optimizer: {e}")
        return False

def main():
    """Main function."""
    
    # Check if optimizer already exists
    if os.path.exists("optimizer_128d_int8.joblib"):
        print("📁 Found existing optimizer file")
        test_result = test_saved_optimizer()
        
        if test_result:
            print("\n🎉 Existing optimizer is working!")
            print("\nNext steps:")
            print("1. export USE_EMBEDDING_OPTIMIZATION=true")
            print("2. python main.py")
            print("3. Upload documents and check dimensions")
            return
        else:
            print("\n⚠️ Existing optimizer has issues, recreating...")
    
    # Create new optimizer
    success = create_and_save_optimizer()
    
    if success:
        print("\n🎉 OPTIMIZER READY!")
        print("\nNext steps:")
        print("1. export USE_EMBEDDING_OPTIMIZATION=true") 
        print("2. python main.py")
        print("3. Upload documents and see 128D embeddings!")
        
        # Test the new optimizer
        test_saved_optimizer()
    else:
        print("\n❌ Failed to create optimizer")

if __name__ == "__main__":
    main()