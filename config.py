from llama_index.embeddings.openai import OpenAIEmbedding
from supabase import create_client
from functools import lru_cache
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.deepseek import DeepSeek
from llama_index.core import Settings
from transformers import AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

# Load environment variables
if not load_dotenv(find_dotenv()):
    print("WARNING: .env file not found.")

# Get environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_CONNECTION = os.getenv("DB_CONNECTION2")
OPENAI_API_KEY=os.getenv("OPENAI_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Logging for verification (remove in production)
logger.info(f"Supabase URL: {SUPABASE_URL[:15]}...")  # Don't log full URL
logger.info("Supabase key present" if SUPABASE_KEY else "Supabase key missing")
logger.info("DB connection present" if DB_CONNECTION else "DB connection missing")

# Initialize shared components
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY, model="text-embedding-3-small", embed_batch_size=10)

@lru_cache(maxsize=5000)
def get_cached_embedding(text: str) -> List[float]:
    """Cache embeddings for repeated text chunks"""
    clean_text = text.strip()[:10000]  # Safety limit
    return embed_model.get_text_embedding(clean_text)

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
DB_CONNECTION = os.getenv("DB_CONNECTION2")


# LLM Config

# LLM Configuration
Settings.llm = DeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.3,
    max_tokens=750,
    timeout=30,
    max_retries=1,
    max_parallel_requests=3,
    metadata={"is_function_calling_model": True}
)
llm = Settings.llm
model_name = "deepseek-ai/DeepSeek-V2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)