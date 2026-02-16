
import logging
import sys
import os
from dotenv import load_dotenv, find_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

# Debug: Print available env var names related to supabase
print("DEBUG: Available SUPABASE env vars:", [k for k in os.environ.keys() if "SUPABASE" in k])

def verify_config():
    logger.info("Starting verification of DB-based LLM configuration...")
    
    try:
        import db_config
        from llm_provider import LLMProviderFactory
        from llama_index.core import Settings
        
        # 1. Verify DB Config Fetching
        logger.info("Fetching LLM config from DB...")
        config = db_config.get_llm_config()
        
        if not config:
            logger.error("FAILED: Could not fetch any active LLM config from DB.")
            
            # DEBUG: Inspect DB Content
            logger.info("--- DEBUGGING DATA ---")
            try:
                providers = db_config.supabase.table("llm_providers").select("id, name, api_key").execute()
                logger.info(f"LLM Providers found: {len(providers.data)}")
                for p in providers.data:
                    logger.info(f"  - Provider: {p.get('name')}, API Key ID: {p.get('api_key')}")
                
                keys = db_config.supabase.table("api_keys").select("id, key_name").execute()
                logger.info(f"API Keys found: {len(keys.data)}")
                for k in keys.data:
                    logger.info(f"  - Key: {k.get('key_name')}, ID: {k.get('id')}")
            except Exception as e:
                logger.error(f"Error inspecting DB: {e}")
                
            return False
            
        logger.info(f"Successfully fetched config for provider: {config.get('provider')}")
        logger.info(f"Model: {config.get('model')}")
        
        if not config.get("api_key"):
            logger.error("FAILED: API key is missing in the fetched config.")
            return False
            
        logger.info("API Key is present (masked): " + "*" * 10)
        
        # 2. Verify settings in config.py
        logger.info("Checking config.py initialization...")
        import config as app_config
        
        if app_config.llm:
            logger.info("Successfully initialized Settings.llm in config.py")
            # Check if it matches somewhat
            current_llm = app_config.llm
            logger.info(f"Current LLM Metadata: {current_llm.metadata.model_name}")
        else:
            logger.error("FAILED: config.llm is None")
            return False
            
        # 3. Verify Embeddings
        if app_config.embed_model:
            logger.info("Successfully initialized embed_model in config.py")
        else:
            logger.warning("Embed model not initialized (this might be expected if no OpenAI key in DB yet)")
            
        return True
        
    except ImportError as e:
        logger.error(f"Import Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Verification Failed with error: {e}")
        # Print stack trace
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_config()
    if success:
        logger.info("✅ VERIFICATION PASSED")
        sys.exit(0)
    else:
        logger.error("❌ VERIFICATION FAILED")
        sys.exit(1)
