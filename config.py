from llama_index.embeddings.openai import OpenAIEmbedding
from functools import lru_cache
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings
from transformers import AutoTokenizer
import logging
import os
import db_config
from llm_provider import LLMProviderFactory

logger = logging.getLogger(__name__)


# Load environment variables (mostly for other configs, keys are in DB now)
if not load_dotenv(find_dotenv()):
    print("WARNING: .env file not found.")

# Get environment variables (Legacy/Other)
DB_CONNECTION = os.getenv("DB_CONNECTION2")

# Logging for verification (remove in production)
logger.info("DB connection present" if DB_CONNECTION else "DB connection missing")

# Initialize shared components
# Embeddings: Try to get from DB, fallback to env for now if needed, but per request we should use DB.
# For now, let's assume OpenAI embedding is standard unless configured otherwise.
# We need to get the API key from DB config.
embedding_config = db_config.get_llm_config(provider_name="openai")
if embedding_config and embedding_config.get("api_key"):
    embed_model = OpenAIEmbedding(
        api_key=embedding_config["api_key"],
        model="text-embedding-3-small",
        embed_batch_size=10
    )
else:
    logger.error("Could not load OpenAI API key from database for embeddings.")
    # Fallback to empty/error state or env if absolutely necessary during migration
    # embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_KEY"), ...)
    # STARTING STRICT: NO ENV FALLBACK AS REQUESTED
    embed_model = None

@lru_cache(maxsize=5000)
def get_cached_embedding(text: str) -> List[float]:
    """Cache embeddings for repeated text chunks"""
    if not embed_model:
         logger.error("Embedding model not initialized.")
         return []
    clean_text = text.strip()[:10000]  # Safety limit
    return embed_model.get_text_embedding(clean_text)

# Supabase client - Re-export from db_config to maintain compatibility
supabase = db_config.supabase
DB_CONNECTION = os.getenv("DB_CONNECTION2")


# LLM Configuration
# Fetch default/active LLM from DB
llm_config = db_config.get_default_llm_config()

if llm_config:
    logger.info(f"Configuring system with LLM: {llm_config.get('provider')} - {llm_config.get('model')}")
    try:
        Settings.llm = LLMProviderFactory.get_llm_instance(
            provider=llm_config.get("provider"),
            api_key=llm_config.get("api_key"),
            model=llm_config.get("model"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 750),
            # Pass other config params if needed
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM from DB config: {e}")
        # Handle failure (maybe set a dummy or raise)
else:
    logger.error("No active LLM configuration found in database.")

llm = Settings.llm
# Tokenizer setup - this might need adjustment if model changes dynamically
# For now, keep the deepseek tokenizer as default or try to match model
model_name = llm_config.get("model") if llm_config else "deepseek-ai/DeepSeek-V2.5"
# fallback for tokenizer name if it's a simple model name like 'gpt-4'
if "gpt" in model_name or "gemini" in model_name:
    # Just use a generic one or don't init this specifics
    pass 

# Keep the original tokenizer loading for safety/legacy
try:
    if "deepseek" in model_name.lower():
         tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2.5", trust_remote_code=True)
    else:
         tokenizer = AutoTokenizer.from_pretrained("gpt2") # Fallback
except Exception as e:
    logger.warning(f"Could not load specific tokenizer: {e}")
    tokenizer = None