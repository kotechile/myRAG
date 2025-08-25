"""
LLM Provider Factory and Adapters

This module provides a factory for creating LLM instances for different providers.
Supported providers: DeepSeek, Gemini, OpenAI, and Anthropic Claude.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
DEFAULT_LLM_CONFIG = {
    "temperature": 0.7,       # More creative/verbose (0-1)
    "max_tokens": 2000,       # Longer responses
    "timeout": 30,
    # Add other default params here
}
# Configure logging
logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """Base class for LLM adapters."""
    
    @abstractmethod
    def get_llm_instance(self):
        """Get the LLM instance."""
        pass
    
    @abstractmethod
    def get_tokenizer(self):
        """Get the appropriate tokenizer for this LLM."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass


class DeepSeekAdapter(LLMAdapter):
    """Adapter for DeepSeek LLM - Enhanced for verbose responses."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
        self._tokenizer = None
        
        # Import DeepSeek
        try:
            from llama_index.llms.deepseek import DeepSeek
            self.DeepSeek = DeepSeek
        except ImportError:
            try:
                from llama_index.llms.deepseek import DeepSeek
                self.DeepSeek = DeepSeek
            except ImportError:
                logger.error("DeepSeek module not found. Please install llama-index with DeepSeek support.")
                self.DeepSeek = None
    
    def get_llm_instance(self):
        if self.DeepSeek is None:
            raise ImportError("DeepSeek module not found. Please install llama-index with DeepSeek support.")
            
        return self.DeepSeek(
            model=self.model,
            api_key=self.api_key,
            temperature=self.kwargs.get("temperature", DEFAULT_LLM_CONFIG["temperature"]),
            max_tokens=self.kwargs.get("max_tokens", DEFAULT_LLM_CONFIG["max_tokens"]),
            timeout=self.kwargs.get("timeout", DEFAULT_LLM_CONFIG["timeout"]),
            max_retries=self.kwargs.get("max_retries", 2),
            max_parallel_requests=self.kwargs.get("max_parallel_requests", 3),
            metadata={"is_function_calling_model": True},
            additional_kwargs={
                "top_p": self.kwargs.get("top_p", 0.95),
                "frequency_penalty": self.kwargs.get("frequency_penalty", 0.1),
                "presence_penalty": self.kwargs.get("presence_penalty", 0.1),
            }
        )
    
    def get_tokenizer(self):
        """Get the appropriate tokenizer for DeepSeek."""
        if self._tokenizer is None:
            try:
                # DeepSeek doesn't have a public tokenizer, use GPT-2 as approximation
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logger.info("Using GPT-2 tokenizer as approximation for DeepSeek")
            except Exception as e:
                logger.error(f"Error loading DeepSeek approximation tokenizer: {str(e)}")
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text using approximation."""
        tokenizer = self.get_tokenizer()
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Error counting tokens with tokenizer: {str(e)}")
                # Fallback to character-based estimation
                return len(text) // 4
        else:
            # Fallback: DeepSeek is roughly 4 characters per token
            return len(text) // 4


class GeminiAdapter(LLMAdapter):
    """Adapter for Google's Gemini LLM."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
        self._tokenizer = None
        logger.info(f"GEMINI ADAPTER SELECTED")
        # Import Gemini
        try:
            from llama_index.llms.gemini import Gemini
            self.Gemini = Gemini
        except ImportError:
            try:
                # Try alternative import path
                from llama_index.llms.gemini import Gemini
                self.Gemini = Gemini
            except ImportError:
                logger.error("Gemini module not found. Please install with: pip install llama-index-llms-gemini")
                self.Gemini = None
    
    def get_llm_instance(self):
        if self.Gemini is None:
            raise ImportError("Gemini module not found. Please install with: pip install llama-index-llms-gemini")
            
        return self.Gemini(
            api_key=self.api_key,
            model=self.model,
            temperature=self.kwargs.get("temperature", DEFAULT_LLM_CONFIG["temperature"]),
            max_tokens=self.kwargs.get("max_tokens", DEFAULT_LLM_CONFIG["max_tokens"]),
            timeout=self.kwargs.get("timeout", DEFAULT_LLM_CONFIG["timeout"]),
            additional_kwargs={"safety_settings": self.kwargs.get("safety_settings", None)}
        )
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            # Gemini doesn't have an official HF tokenizer, so we'll use an approximation
            try:
                from transformers import AutoTokenizer
                # T5 tokenizer is often used as an approximation for Gemini
                self._tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            except Exception as e:
                logger.error(f"Error loading Gemini approximation tokenizer: {str(e)}")
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Fallback: rough estimation based on characters (Gemini is roughly 4 chars per token)
            return len(text) // 4


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI LLMs."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
        self._tokenizer = None
        
        # Import OpenAI
        try:
            from llama_index.llms.openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            try:
                # Try alternative import path
                from llama_index.llms.openai import OpenAI
                self.OpenAI = OpenAI
            except ImportError:
                logger.error("OpenAI module not found. Please ensure llama-index is installed correctly.")
                self.OpenAI = None
        
    def get_llm_instance(self):
        if self.OpenAI is None:
            raise ImportError("OpenAI module not found. Please ensure llama-index is installed correctly.")
            
        return self.OpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=self.kwargs.get("temperature", DEFAULT_LLM_CONFIG["temperature"]),
            max_tokens=self.kwargs.get("max_tokens", DEFAULT_LLM_CONFIG["max_tokens"]),
            timeout=self.kwargs.get("timeout", DEFAULT_LLM_CONFIG["timeout"]),
            max_retries=self.kwargs.get("max_retries", 1),
            additional_kwargs=self.kwargs.get("additional_kwargs", {})
        )
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            try:
                # Use tiktoken if available
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except (ImportError, Exception) as e:
                logger.error(f"Error loading tiktoken for OpenAI: {str(e)}")
                # Fallback to GPT-2 tokenizer from HF which is close to OpenAI's
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
                except Exception as e2:
                    logger.error(f"Error loading fallback tokenizer: {str(e2)}")
                    self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.get_tokenizer()
        if tokenizer:
            if hasattr(tokenizer, 'encode'):
                return len(tokenizer.encode(text))
            elif hasattr(tokenizer, 'tokenize'):
                return len(tokenizer.tokenize(text))
        # Fallback: rough estimation based on GPT average
        return len(text) // 4


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude LLMs."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
        self._tokenizer = None
        
        # Import Anthropic
        try:
            from llama_index.llms.anthropic import Anthropic
            self.Anthropic = Anthropic
        except ImportError:
            try:
                # Try alternative import path
                from llama_index.llms.anthropic import Anthropic
                self.Anthropic = Anthropic
            except ImportError:
                logger.error("Anthropic module not found. Please install with: pip install llama-index-llms-anthropic")
                self.Anthropic = None
    
    def get_llm_instance(self):
        if self.Anthropic is None:
            raise ImportError("Anthropic module not found. Please install with: pip install llama-index-llms-anthropic")
            
        return self.Anthropic(
            api_key=self.api_key,
            model=self.model,
            temperature=self.kwargs.get("temperature", DEFAULT_LLM_CONFIG["temperature"]),
            max_tokens=self.kwargs.get("max_tokens", DEFAULT_LLM_CONFIG["max_tokens"]),
            timeout=self.kwargs.get("timeout", DEFAULT_LLM_CONFIG["timeout"]),
            additional_kwargs=self.kwargs.get("additional_kwargs", {})
        )
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            # Claude doesn't have a public tokenizer, use GPT-2 as approximate
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                logger.error(f"Error loading Claude approximation tokenizer: {str(e)}")
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Fallback: Claude is roughly 4 chars per token
            return len(text) // 4


class LLMProviderFactory:
    """Factory for creating LLM instances."""
    
    _instances = {}
    _adapters = {
        "deepseek": DeepSeekAdapter,
        "gemini": GeminiAdapter,
        "openai": OpenAIAdapter,
        "claude": AnthropicAdapter
    }
    
    @classmethod
    def get_adapter(cls, provider: str, **kwargs) -> LLMAdapter:
        """
        Get an LLM adapter for the specified provider.
        
        Args:
            provider: Name of the LLM provider (deepseek, gemini, openai, claude)
            **kwargs: Additional arguments to pass to the adapter
            
        Returns:
            LLMAdapter instance
        """
        provider = provider.lower()
        
        if provider not in cls._adapters:
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           f"Supported providers: {', '.join(cls._adapters.keys())}")
        
        # Get API key from environment or kwargs
        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            # Try to get from environment
            env_var = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(env_var)
            
            if api_key is None:
                raise ValueError(f"API key not provided for {provider}. "
                               f"Set {env_var} environment variable or pass api_key in kwargs.")
        
        # Create and return adapter
        adapter_cls = cls._adapters[provider]
        return adapter_cls(api_key=api_key, **kwargs)
    
    @classmethod
    def get_llm_instance(cls, provider: str, **kwargs):
        """
        Get an LLM instance for the specified provider.
        
        Args:
            provider: Name of the LLM provider (deepseek, gemini, openai, claude)
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            LLM instance
        """
        adapter = cls.get_adapter(provider, **kwargs)
        return adapter.get_llm_instance()
