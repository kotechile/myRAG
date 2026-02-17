
import os
import logging
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
# Prefer service role key for backend operations if available
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# Clean credentials
if SUPABASE_URL:
    SUPABASE_URL = SUPABASE_URL.strip("'\" \n\t")
if SUPABASE_KEY:
    SUPABASE_KEY = SUPABASE_KEY.strip("'\" \n\t")

# Validate credentials before attempting initialization
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error(f"Supabase credentials missing or empty. URL present: {bool(SUPABASE_URL)}, Key present: {bool(SUPABASE_KEY)}")
    supabase = None
else:
    # Create Supabase client
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Supabase client initialized successfully. URL: {SUPABASE_URL[:20]}...")
    except Exception as e:
        logger.error(f"🚨 Failed to initialize Supabase client: {str(e)}")
        logger.error(f"Details - URL: {SUPABASE_URL[:20]}..., Key Length: {len(SUPABASE_KEY) if SUPABASE_KEY else 0}")
        supabase = None

def get_llm_config(provider_name: str = None, model_name: str = None) -> Optional[Dict[str, Any]]:
    """
    Fetch LLM configuration and API key from the database.
    
    Args:
        provider_name: Optional provider name filter (e.g., 'google', 'openai')
        model_name: Optional model name filter
        
    Returns:
        Dictionary with configuration or None if not found/error
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return None

    try:
        # Query llm_providers
        # We want active providers.
        query = supabase.table("llm_providers").select("*").eq("is_active", True)

        if provider_name:
            query = query.ilike("provider", provider_name)
        
        if model_name:
            query = query.eq("model_name", model_name)
            
        # Order by priority or is_default if no specific model requested
        query = query.order("priority", desc=True).order("is_default", desc=True)
        
        result = query.execute()
        
        if not result.data:
            logger.warning(f"No active LLM provider found matching: provider={provider_name}, model={model_name}")
            return None
            
        # Iterate through providers to find one with a valid key
        for provider_config in result.data:
            # Extract API key ID (Fixed: column is api_keys_id, not api_key)
            api_key_id = provider_config.get("api_keys_id")
            api_key_data = None
            
            if api_key_id:
                # 1. Try fetching by ID
                try:
                    key_result = supabase.table("api_keys").select("*").eq("id", api_key_id).execute()
                    if key_result.data:
                        api_key_data = key_result.data[0]
                except Exception as e_id:
                    logger.warning(f"Error fetching key by ID {api_key_id}: {e_id}")
            
            if not api_key_data:
                # 2. Fallback: Try fetching by provider name
                provider_type = provider_config.get("provider")
                if provider_type:
                    # Map 'google' to 'gemini' if needed (based on api_keys screenshot)
                    provider_search = provider_type.lower()
                    if provider_search == "google":
                        provider_search = "gemini"
                    
                    try:
                        key_result = supabase.table("api_keys").select("*").ilike("provider", provider_search).eq("is_active", True).execute()
                        if key_result.data:
                            # Use the first active key found for this provider
                            api_key_data = key_result.data[0]
                    except Exception as e_name:
                        logger.warning(f"Error fetching key by provider name {provider_search}: {e_name}")

            if not api_key_data or not api_key_data.get("key_value"):
                logger.warning(f"No valid API key found for provider {provider_config.get('name')}. provider_config: {provider_config}. api_key_data: {api_key_data}")
                continue
            
            logger.info(f"Successfully resolved key for {provider_config.get('name')} using {'ID' if api_key_id and api_key_data.get('id') == api_key_id else 'fallback'}")

                    
            # Found a valid config!
            config = {
                "provider": provider_config.get("provider"), # Fixed: column is provider, not provider_type
                "model": provider_config.get("model_name"),
                "api_key": api_key_data.get("key_value"),
                "temperature": provider_config.get("temperature"),
                "max_tokens": provider_config.get("max_tokens"),
                "top_p": provider_config.get("top_p"),
                "frequency_penalty": provider_config.get("frequency_penalty"),
                "presence_penalty": provider_config.get("presence_penalty"),
                # Add other fields as needed
            }
            
            # Filter out None values
            return {k: v for k, v in config.items() if v is not None}

        logger.error("No valid LLM configuration could be resolved from active providers.")
        return None

    except Exception as e:
        logger.error(f"Error fetching LLM config from database: {e}")
        return None

def get_default_llm_config() -> Optional[Dict[str, Any]]:
    """Helper to get the default active LLM configuration."""
    return get_llm_config()


def get_api_key(provider_name: str) -> Optional[str]:
    """
    Fetch API key for a specific provider from the api_keys table.
    
    Args:
        provider_name: Name of the provider (case-insensitive)
        
    Returns:
        API key string or None if not found
    """
    if not supabase:
        logger.error("Supabase client is not initialized.")
        return None
        
    try:
        # First try matching 'provider' column
        result = supabase.table("api_keys").select("key_value").ilike("provider", provider_name).execute()
        if result.data:
            return result.data[0].get("key_value")
            
        # Fallback to matching 'name' column
        result = supabase.table("api_keys").select("key_value").ilike("name", f"%{provider_name}%").execute()
        if result.data:
            return result.data[0].get("key_value")
            
        return None
        
    except Exception as e:
        logger.error(f"Error fetching API key for {provider_name}: {e}")
        return None
