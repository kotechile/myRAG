# embedding_optimizer.py - Complete embedding optimization implementation

import os
import logging
import joblib
import numpy as np
from typing import List, Optional, Union, Any, TYPE_CHECKING
from pydantic import Field
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QuantizedTransformer(BaseEstimator, TransformerMixin):
    """Quantize embeddings to int8 to save space."""
    
    def __init__(self, n_bits: int = 8):
        self.n_bits = n_bits
        self.scale_ = None
        self.zero_point_ = None
        self.fitted = False
        
    def fit(self, X: np.ndarray, y=None):
        """Learn quantization parameters from data."""
        X = np.array(X)
        
        # Calculate quantization parameters
        x_min = X.min()
        x_max = X.max()
        
        # Calculate scale and zero point for quantization
        if self.n_bits == 8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 2**self.n_bits - 1
            
        self.scale_ = (x_max - x_min) / (qmax - qmin)
        self.zero_point_ = qmin - x_min / self.scale_
        
        self.fitted = True
        logger.info(f"✅ Quantization fitted: scale={self.scale_:.6f}, zero_point={self.zero_point_:.6f}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Quantize the embeddings."""
        if not self.fitted:
            raise ValueError("Quantizer must be fitted before transform")
            
        X = np.array(X)
        
        # Quantize
        X_quantized = X / self.scale_ + self.zero_point_
        
        if self.n_bits == 8:
            X_quantized = np.clip(X_quantized, -128, 127).astype(np.int8)
        else:
            X_quantized = np.clip(X_quantized, 0, 2**self.n_bits - 1).astype(np.uint8)
            
        return X_quantized
    
    def inverse_transform(self, X_quantized: np.ndarray) -> np.ndarray:
        """Dequantize back to float."""
        if not self.fitted:
            raise ValueError("Quantizer must be fitted before inverse_transform")
            
        X_quantized = np.array(X_quantized, dtype=np.float32)
        X_dequantized = (X_quantized - self.zero_point_) * self.scale_
        return X_dequantized


class CombinedOptimizer(BaseEstimator, TransformerMixin):
    """Combined PCA + Quantization optimizer."""
    
    def __init__(self, target_dim: int = 128, quantization: str = "int8"):
        self.target_dim = target_dim
        self.quantization = quantization
        self.pca = PCA(n_components=target_dim)
        self.quantizer = QuantizedTransformer(n_bits=8 if quantization == "int8" else 16)
        self.fitted = False
        
    def fit(self, X: np.ndarray, y=None):
        """Fit both PCA and quantization."""
        X = np.array(X)
        logger.info(f"🔧 Fitting optimizer on {X.shape} data...")
        
        # Fit PCA first
        logger.info(f"📐 Fitting PCA: {X.shape[1]} → {self.target_dim} dimensions")
        X_pca = self.pca.fit_transform(X)
        
        # Then fit quantizer
        logger.info(f"🔢 Fitting quantizer: {self.quantization}")
        self.quantizer.fit(X_pca)
        
        self.fitted = True
        logger.info(f"✅ CombinedOptimizer fitted successfully")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA + quantization."""
        if not self.fitted:
            raise ValueError("Optimizer must be fitted before transform")
            
        X = np.array(X)
        
        # Apply PCA
        X_pca = self.pca.transform(X)
        
        # Apply quantization
        X_quantized = self.quantizer.transform(X_pca)
        
        return X_quantized
    
    def inverse_transform(self, X_optimized: np.ndarray) -> np.ndarray:
        """Reverse the optimization (approximate)."""
        if not self.fitted:
            raise ValueError("Optimizer must be fitted before inverse_transform")
            
        # Dequantize
        X_dequantized = self.quantizer.inverse_transform(X_optimized)
        
        # Inverse PCA
        X_reconstructed = self.pca.inverse_transform(X_dequantized)
        
        return X_reconstructed
    
    def save(self, filepath: str):
        """Save the fitted optimizer."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted optimizer")
            
        joblib.dump(self, filepath)
        logger.info(f"💾 Optimizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a saved optimizer."""
        optimizer = joblib.load(filepath)
        logger.info(f"📁 Optimizer loaded from {filepath}")
        return optimizer


from typing import List, Optional, Union, Any, TYPE_CHECKING
from pydantic import Field

if TYPE_CHECKING:
    pass

class OptimizedOpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model with optimization that properly inherits from BaseEmbedding."""
    
    # Properly declare Pydantic fields (no leading underscores)
    optimizer: Optional[CombinedOptimizer] = Field(default=None, exclude=True)
    base_model_instance: Optional[OpenAIEmbedding] = Field(default=None, exclude=True)
    dimension_cache: Optional[int] = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = "forbid"
    
    def __init__(self, 
                 model: str = "text-embedding-3-small",
                 optimizer: Optional[CombinedOptimizer] = None,
                 embed_batch_size: int = 10,
                 **kwargs):
        
        # Create the base model
        base_model = OpenAIEmbedding(
            model=model, 
            embed_batch_size=embed_batch_size,
            **kwargs
        )
        
        # Initialize the parent class properly
        super().__init__(
            model_name=model,
            embed_batch_size=embed_batch_size,
            optimizer=optimizer,
            base_model_instance=base_model,
            dimension_cache=None,
            **kwargs
        )
        
        logger.info(f"🔧 OptimizedOpenAIEmbedding created with model: {model}")
        if optimizer and optimizer.fitted:
            logger.info(f"✅ Using fitted optimizer: {optimizer.target_dim}D + {optimizer.quantization}")
        elif optimizer:
            logger.info(f"⚠️ Optimizer attached but not fitted yet")
        else:
            logger.info(f"📊 No optimizer - using standard embeddings")
    
    @property
    def base_model(self) -> OpenAIEmbedding:
        """Get the base OpenAI embedding model."""
        return self.base_model_instance
    
    def _apply_optimization(self, embeddings: Union[List[float], List[List[float]]]) -> Union[List[float], List[List[float]]]:
        """Apply optimization to embeddings if available."""
        if not self.optimizer or not self.optimizer.fitted:
            return embeddings
            
        try:
            # Handle single embedding
            if isinstance(embeddings[0], (int, float)):
                embedding_array = np.array(embeddings).reshape(1, -1)
                optimized = self.optimizer.transform(embedding_array)
                return optimized.flatten().tolist()
            # Handle batch embeddings
            else:
                embeddings_array = np.array(embeddings)
                optimized = self.optimizer.transform(embeddings_array)
                return optimized.tolist()
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return embeddings
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get optimized embedding for single query."""
        embedding = self.base_model._get_query_embedding(query)
        return self._apply_optimization(embedding)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get optimized embedding for single text."""
        embedding = self.base_model._get_text_embedding(text)
        return self._apply_optimization(embedding)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get optimized embeddings for batch of texts."""
        embeddings = self.base_model._get_text_embeddings(texts)
        return self._apply_optimization(embeddings)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding."""
        embedding = await self.base_model._aget_query_embedding(query)
        return self._apply_optimization(embedding)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of get_text_embedding."""
        embedding = await self.base_model._aget_text_embedding(text)
        return self._apply_optimization(embedding)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of get_text_embeddings."""
        embeddings = await self.base_model._aget_text_embeddings(texts)
        return self._apply_optimization(embeddings)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get optimized embedding for single text."""
        embedding = self.base_model.get_text_embedding(text)
        return self._apply_optimization(embedding)
    
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get optimized embeddings for batch of texts."""
        embeddings = self.base_model.get_text_embedding_batch(texts)
        return self._apply_optimization(embeddings)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (after optimization)."""
        if self.dimension_cache is None:
            test_embedding = self.get_text_embedding("test")
            self.dimension_cache = len(test_embedding)
        return self.dimension_cache
    
    def fit_optimizer(self, sample_texts: List[str]) -> None:
        """Fit the optimizer using sample texts."""
        if not self.optimizer:
            raise ValueError("No optimizer attached")
            
        logger.info(f"🔧 Fitting optimizer with {len(sample_texts)} sample texts...")
        
        # Generate sample embeddings using base model
        sample_embeddings = []
        for text in sample_texts:
            embedding = self.base_model.get_text_embedding(text)
            sample_embeddings.append(embedding)
        
        # Fit optimizer
        sample_array = np.array(sample_embeddings)
        self.optimizer.fit(sample_array)
        
        # Clear dimension cache
        self.dimension_cache = None
        
        logger.info(f"✅ Optimizer fitted successfully")


def create_128d_int8_optimizer() -> CombinedOptimizer:
    """Create a 128D int8 optimizer."""
    optimizer = CombinedOptimizer(target_dim=128, quantization="int8")
    logger.info("🔧 Created 128D int8 optimizer (unfitted)")
    return optimizer


def create_384d_int8_optimizer() -> CombinedOptimizer:
    """Create a 384D int8 optimizer."""
    optimizer = CombinedOptimizer(target_dim=384, quantization="int8")
    logger.info("🔧 Created 384D int8 optimizer (unfitted)")
    return optimizer


def fit_and_save_optimizer(sample_texts: Optional[List[str]] = None, 
                          target_dim: int = 128,
                          quantization: str = "int8",
                          save_path: str = "optimizer_128d_int8.joblib") -> CombinedOptimizer:
    """Convenience function to fit and save an optimizer."""
    
    logger.info(f"🔧 Creating and fitting optimizer: {target_dim}D + {quantization}")
    
    # Generate enough sample texts if not provided
    if sample_texts is None or len(sample_texts) < target_dim * 2:
        logger.info(f"📝 Generating sample texts (need at least {target_dim * 2} for {target_dim}D PCA)")
        base_texts = [
            "Real estate investment and property management strategies",
            "Financial planning and portfolio optimization techniques", 
            "Machine learning algorithms and data science methods",
            "Natural language processing and text analysis tools",
            "Business development and market research approaches",
            "Software engineering and development best practices",
            "Digital marketing and customer engagement strategies",
            "Data analytics and statistical modeling techniques",
            "Project management and organizational efficiency",
            "Technology innovation and digital transformation"
        ]
        
        sample_texts = []
        variations = target_dim * 2 // len(base_texts) + 1
        
        for i in range(variations):
            for base_text in base_texts:
                sample_texts.append(f"{base_text}. Context variation {i+1} with additional details.")
                if len(sample_texts) >= target_dim * 2:
                    break
            if len(sample_texts) >= target_dim * 2:
                break
        
        sample_texts = sample_texts[:target_dim * 2]  # Limit to exactly what we need
        logger.info(f"📊 Generated {len(sample_texts)} sample texts")
    
    # Create base embedding model
    base_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Generate sample embeddings in batches to avoid rate limits
    logger.info(f"🔄 Generating embeddings for {len(sample_texts)} sample texts...")
    sample_embeddings = []
    
    batch_size = 20  # Process in smaller batches
    for i in range(0, len(sample_texts), batch_size):
        batch = sample_texts[i:i + batch_size]
        batch_embeddings = base_model.get_text_embedding_batch(batch)
        sample_embeddings.extend(batch_embeddings)
        logger.info(f"   Processed batch {i//batch_size + 1}/{(len(sample_texts)-1)//batch_size + 1}")
    
    # Create and fit optimizer
    optimizer = CombinedOptimizer(target_dim=target_dim, quantization=quantization)
    sample_array = np.array(sample_embeddings)
    logger.info(f"📐 Fitting optimizer on {sample_array.shape} embeddings...")
    optimizer.fit(sample_array)
    
    # Save optimizer
    optimizer.save(save_path)
    
    logger.info(f"✅ Optimizer created, fitted, and saved to {save_path}")
    return optimizer


def test_optimization(sample_texts: Optional[List[str]] = None) -> None:
    """Test the optimization pipeline."""
    
    if sample_texts is None:
        # Generate enough sample texts for PCA fitting
        base_texts = [
            "This is a test document for optimization and machine learning.",
            "Another sample text for embedding optimization and neural networks.",
            "Real estate investment strategies and property management techniques.",
            "Machine learning algorithms and artificial intelligence concepts.",
            "Natural language processing and computational text analysis.",
            "Financial planning and investment portfolio management strategies.",
            "Data science methodologies and statistical analysis techniques.",
            "Software development practices and programming best practices.",
            "Business analytics and market research methodologies.",
            "Digital marketing strategies and customer engagement techniques."
        ]
        
        # Expand to have enough samples for 128D PCA
        sample_texts = []
        for i in range(15):  # Create 150 samples
            for base_text in base_texts:
                sample_texts.append(f"{base_text} Sample variation {i+1}.")
        
        logger.info(f"📊 Generated {len(sample_texts)} sample texts for PCA fitting")
    
    logger.info("🧪 Testing optimization pipeline...")
    
    # Test standard embedding
    logger.info("1️⃣ Testing standard embedding:")
    standard_model = OpenAIEmbedding(model="text-embedding-3-small")
    standard_embedding = standard_model.get_text_embedding(sample_texts[0])
    logger.info(f"   Standard dimension: {len(standard_embedding)}")
    logger.info(f"   Sample values: {standard_embedding[:3]}")
    
    # Test optimized embedding
    logger.info("2️⃣ Testing optimized embedding:")
    
    # Create and fit optimizer
    optimizer = create_128d_int8_optimizer()
    optimized_model = OptimizedOpenAIEmbedding(optimizer=optimizer)
    
    # Fit optimizer
    optimized_model.fit_optimizer(sample_texts)
    
    # Test optimized embedding
    optimized_embedding = optimized_model.get_text_embedding(sample_texts[0])
    logger.info(f"   Optimized dimension: {len(optimized_embedding)}")
    logger.info(f"   Sample values: {optimized_embedding[:3]}")
    
    # Compare
    compression_ratio = len(standard_embedding) / len(optimized_embedding)
    logger.info(f"🎯 Compression ratio: {compression_ratio:.1f}x")
    
    # Test batch processing
    logger.info("3️⃣ Testing batch processing:")
    batch_embeddings = optimized_model.get_text_embedding_batch(sample_texts[:3])
    logger.info(f"   Batch size: {len(batch_embeddings)}")
    logger.info(f"   Each embedding dimension: {len(batch_embeddings[0])}")
    
    logger.info("✅ Optimization test completed successfully!")


if __name__ == "__main__":
    # Run test
    test_optimization()