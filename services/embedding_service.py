"""
Embedding Service - Sentence Transformer Embeddings with Persistent Cache
==========================================================================
This module handles text embedding generation with caching to avoid
regenerating embeddings on every application restart.

Features:
- Persistent disk caching of embeddings
- Automatic cache validation
- GPU/CPU support with fallback
- Parallel processing for CPU mode
- Progress tracking and logging
"""

import numpy as np
import torch
import time
import hashlib
import logging
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import multiprocessing


class EmbeddingService:
    """
    Service for generating and caching sentence embeddings.
    
    This service intelligently caches embeddings to disk, avoiding regeneration
    on subsequent runs. It only regenerates when the dataset or model changes.
    Supports both GPU and CPU with automatic optimization.
    
    Attributes:
        config: Model configuration
        cache_manager: Runtime cache manager
        model: SentenceTransformer model instance
        device: Computation device ('cuda' or 'cpu')
        model_lock: Thread lock for model access
    """
    
    def __init__(self, config_service, cache_manager):
        """
        Initialize the EmbeddingService.
        
        Args:
            config_service: Configuration service instance
            cache_manager: Cache manager for runtime caching
        """
        self.config = config_service.get_model_config()
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.device = None
        self.model_lock = threading.RLock()
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the sentence transformer model.
        
        Loads the model from HuggingFace and sets up GPU/CPU accordingly.
        """
        with self.model_lock:
            self.logger.info(f"Loading model: {self.config.model_name}")
            
            self.device = self._setup_device()
            self.logger.info(f"Using device: {self.device}")
            
            start_time = time.time()
            self.model = SentenceTransformer(self.config.model_name)
            
            if self.device == 'cuda':
                self.model.to(self.device)
                torch.cuda.empty_cache()
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    def _setup_device(self) -> str:
        """
        Setup computation device (GPU/CPU).
        
        Returns:
            str: Device name ('cuda' or 'cpu')
        """
        if self.config.use_gpu and torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU: {gpu_name}, VRAM: {gpu_memory:.1f} GB")
        else:
            device = 'cpu'
            torch.set_num_threads(self.config.cpu_cores)
            self.logger.info(f"Using CPU with {self.config.cpu_cores} threads")
        return device
    
    def _get_embeddings_cache_key(self, texts: List[str]) -> str:
        """
        Generate a STABLE cache key for embeddings based on texts and config.
        Uses only stable parameters that won't change between runs.
        
        Args:
            texts: List of texts to generate cache key for
            
        Returns:
            str: MD5 hash of the configuration
        """
        # SADECE stabil parametreleri kullan
        key_parts = [
            self.config.model_name,
            str(len(texts)),
            self.config.dataset_name,  # Dataset adı ekle
            str(self.config.max_faqs) if self.config.max_faqs else "all"
        ]
        key_string = "_".join(key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        self.logger.debug(f"Cache key generated: {cache_key}")
        self.logger.debug(f"Key components: {key_parts}")
        
        return cache_key
    
    def _save_embeddings_to_disk(self, embeddings: np.ndarray, texts: List[str], 
                                  cache_key: str) -> bool:
        """
        Save embeddings to disk cache with improved metadata.
        
        Args:
            embeddings: Numpy array of embeddings
            texts: Original texts (for verification)
            cache_key: Cache identifier
            
        Returns:
            bool: True if save successful
        """
        try:
            cache_file = self.config.embeddings_cache_file
            
            cache_data = {
                'embeddings': embeddings,
                'num_texts': len(texts),
                'embedding_dim': embeddings.shape[1],
                'model_name': self.config.model_name,
                'dataset_name': self.config.dataset_name,
                'max_faqs': self.config.max_faqs,
                'cache_key': cache_key,
                'created_at': time.time(),
                'version': '2.0'  # Cache format versiyonu
            }
            
            self.logger.info(f"Saving embeddings to disk: {cache_file}")
            self.logger.info(f"Cache key being saved: {cache_key}")
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(cache_file) / (1024 * 1024)
            self.logger.info(f"✓ Embeddings saved: {file_size:.1f} MB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save embeddings to disk: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_embeddings_from_disk(self, texts: List[str], 
                                     cache_key: str) -> Optional[np.ndarray]:
        """
        Load embeddings from disk cache if available and valid.
        IMPROVED: More lenient validation for stability.
        
        Args:
            texts: Current texts to verify against
            cache_key: Expected cache identifier
            
        Returns:
            Optional[np.ndarray]: Cached embeddings or None if not available/invalid
        """
        try:
            cache_file = self.config.embeddings_cache_file
            
            if not os.path.exists(cache_file):
                self.logger.info("No embedding cache found on disk")
                return None
            
            if self.config.force_regenerate_embeddings:
                self.logger.info("Force regenerate flag set, ignoring cache")
                return None
            
            self.logger.info(f"Loading embeddings from disk: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 1. Cache key kontrolü
            cached_key = cache_data.get('cache_key')
            if cached_key != cache_key:
                self.logger.warning(f"Cache key mismatch:")
                self.logger.warning(f"  Expected: {cache_key}")
                self.logger.warning(f"  Found: {cached_key}")
                return None
            
            # 2. Model kontrolü
            if cache_data.get('model_name') != self.config.model_name:
                self.logger.warning(f"Model changed: {cache_data.get('model_name')} -> {self.config.model_name}")
                return None
            
            # 3. Text sayısı kontrolü (daha esnek - %5 tolerans)
            cached_count = cache_data.get('num_texts', 0)
            current_count = len(texts)
            
            tolerance = max(10, int(current_count * 0.05))
            if abs(cached_count - current_count) > tolerance:
                self.logger.warning(f"Text count significantly changed: {cached_count} -> {current_count} (tolerance: {tolerance})")
                return None
            
            # 4. Dataset kontrolü
            if cache_data.get('dataset_name') != self.config.dataset_name:
                self.logger.warning(f"Dataset changed: {cache_data.get('dataset_name')} -> {self.config.dataset_name}")
                return None
            
            # 5. Embedding boyutu kontrolü
            embeddings = cache_data['embeddings']
            if embeddings.shape[0] != current_count:
                self.logger.warning(f"Embedding count mismatch: {embeddings.shape[0]} != {current_count}")
                # Eğer fark küçükse, mevcut sayı kadar kullan
                if embeddings.shape[0] > current_count:
                    self.logger.info(f"Truncating embeddings from {embeddings.shape[0]} to {current_count}")
                    embeddings = embeddings[:current_count]
                else:
                    return None
            
            file_size = os.path.getsize(cache_file) / (1024 * 1024)
            cache_age = (time.time() - cache_data.get('created_at', 0)) / 3600
            
            self.logger.info(f"✓ Loaded cached embeddings: {embeddings.shape}")
            self.logger.info(f"  Cache size: {file_size:.1f} MB")
            self.logger.info(f"  Cache age: {cache_age:.1f} hours")
            self.logger.info(f"  Model: {cache_data.get('model_name')}")
            self.logger.info(f"  Dataset: {cache_data.get('dataset_name')}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings from disk: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_cache_key(self, text: str, normalize: bool = True) -> str:
        """
        Generate cache key for a single text.
        
        Args:
            text: Input text
            normalize: Whether normalization was applied
            
        Returns:
            str: MD5 hash cache key
        """
        content = f"{text}_{self.config.model_name}_{normalize}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def encode_single(self, text: str, normalize: bool = True, 
                     use_cache: bool = True) -> np.ndarray:
        """
        Encode a single text to embedding vector.
        
        Args:
            text: Input text to encode
            normalize: Whether to normalize the embedding
            use_cache: Whether to use runtime cache
            
        Returns:
            np.ndarray: Embedding vector
        """
        if use_cache:
            cache_key = self._get_cache_key(text, normalize)
            cached_result = self.cache_manager.get_embedding(cache_key)
            if cached_result is not None:
                return cached_result
        
        with self.model_lock:
            embedding = self.model.encode(
                [text],
                convert_to_tensor=False,
                convert_to_numpy=True,
                device='cpu',
                normalize_embeddings=normalize,
                show_progress_bar=False
            )[0]
        
        if use_cache:
            self.cache_manager.set_embedding(cache_key, embedding)
        
        return embedding
    
    def encode_batch(self, texts: List[str], normalize: bool = True, 
                    use_cache: bool = True) -> np.ndarray:
        """
        Encode a batch of texts to embedding vectors with intelligent disk caching.
        
        This method first checks for a complete embedding cache on disk. If found
        and valid, it loads the entire batch instantly. Otherwise, it generates
        embeddings and saves them for future use.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings
            use_cache: Whether to use caching (both disk and runtime)
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        # Generate cache key for this batch
        cache_key = self._get_embeddings_cache_key(texts)
        
        # Try to load from disk cache first
        if use_cache:
            cached_embeddings = self._load_embeddings_from_disk(texts, cache_key)
            if cached_embeddings is not None:
                self.logger.info("✓ Using cached embeddings from disk - skipping generation!")
                return cached_embeddings
        
        # Cache miss - need to generate embeddings
        self.logger.info(f"Generating embeddings for {len(texts):,} texts...")
        self.logger.info("This may take a while on first run, but will be cached for future use")
        
        # Check runtime cache for individual texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                runtime_key = self._get_cache_key(text, normalize)
                cached_result = self.cache_manager.get_embedding(runtime_key)
                if cached_result is not None:
                    cached_embeddings[i] = cached_result
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            cache_hit_rate = (1 - len(uncached_texts) / len(texts)) * 100
            self.logger.info(f"Runtime cache hit rate: {cache_hit_rate:.1f}%")
            self.logger.info(f"Encoding {len(uncached_texts):,} uncached texts...")
            
            optimal_batch = self._get_optimal_batch_size()
            
            if self.device == 'cuda':
                new_embeddings = self._encode_gpu_batch(uncached_texts, optimal_batch, normalize)
            else:
                new_embeddings = self._encode_cpu_batch(uncached_texts, optimal_batch, normalize)
            
            # Cache individual embeddings
            if use_cache:
                for i, text in enumerate(uncached_texts):
                    runtime_key = self._get_cache_key(text, normalize)
                    self.cache_manager.set_embedding(runtime_key, new_embeddings[i])
        else:
            new_embeddings = np.array([])
            self.logger.info("All embeddings found in runtime cache!")
        
        # Combine cached and newly generated embeddings
        final_embeddings = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        
        for i, embedding in cached_embeddings.items():
            final_embeddings[i] = embedding
        
        for i, uncached_idx in enumerate(uncached_indices):
            final_embeddings[uncached_idx] = new_embeddings[i]
        
        # Save complete embedding set to disk for future use
        if use_cache:
            self._save_embeddings_to_disk(final_embeddings, texts, cache_key)
        
        return final_embeddings
    
    def _encode_gpu_batch(self, texts: List[str], batch_size: int, 
                         normalize: bool) -> np.ndarray:
        """
        Encode texts using GPU.
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize
            
        Returns:
            np.ndarray: Embeddings array
        """
        with self.model_lock:
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=False,
                    convert_to_numpy=True,
                    device=self.device,
                    normalize_embeddings=normalize
                )
                torch.cuda.empty_cache()
                return embeddings
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning("GPU OOM - falling back to CPU")
                    torch.cuda.empty_cache()
                    self.device = 'cpu'
                    self.model.to('cpu')
                    return self._encode_cpu_batch(texts, batch_size // 2, normalize)
                else:
                    raise e
    
    def _encode_cpu_batch(self, texts: List[str], batch_size: int, 
                         normalize: bool) -> np.ndarray:
        """
        Encode texts using CPU with parallel processing.
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize
            
        Returns:
            np.ndarray: Embeddings array
        """
        def encode_batch_worker(batch):
            with self.model_lock:
                return self.model.encode(
                    batch,
                    convert_to_tensor=False,
                    convert_to_numpy=True,
                    device='cpu',
                    show_progress_bar=False,
                    normalize_embeddings=normalize
                )
        
        num_workers = min(self.config.embedding_threads, multiprocessing.cpu_count())
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        all_embeddings = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {executor.submit(encode_batch_worker, batch): i 
                              for i, batch in enumerate(batches)}
            
            for future in tqdm(as_completed(future_to_batch), total=len(batches), 
                             desc="CPU Encoding"):
                batch_idx = future_to_batch[future]
                batch_embeddings = future.result()
                all_embeddings.append((batch_idx, batch_embeddings))
        
        all_embeddings.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in all_embeddings])
    
    def _get_optimal_batch_size(self) -> int:
        """
        Determine optimal batch size for encoding.
        
        Returns:
            int: Optimal batch size
        """
        if self.device == 'cpu':
            return self.config.batch_size
        
        test_batch_sizes = [256, 128, 64, 32]
        
        for batch_size in test_batch_sizes:
            try:
                test_texts = ["test text"] * min(batch_size, 10)
                with self.model_lock:
                    self.model.encode(
                        test_texts,
                        batch_size=batch_size,
                        convert_to_tensor=False,
                        convert_to_numpy=True,
                        device=self.device,
                        show_progress_bar=False
                    )
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                self.logger.debug(f"Optimal batch size determined: {batch_size}")
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return 16
    
    def clear_embeddings_cache(self) -> bool:
        """
        Clear the disk embeddings cache file.
        
        Returns:
            bool: True if successful
        """
        try:
            cache_file = self.config.embeddings_cache_file
            if os.path.exists(cache_file):
                os.remove(cache_file)
                self.logger.info(f"Embeddings cache cleared: {cache_file}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to clear embeddings cache: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and cache status.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        cache_info = {}
        cache_file = self.config.embeddings_cache_file
        
        if os.path.exists(cache_file):
            cache_size = os.path.getsize(cache_file) / (1024 * 1024)
            cache_modified = os.path.getmtime(cache_file)
            cache_age = (time.time() - cache_modified) / 3600
            
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                cache_info = {
                    'cache_exists': True,
                    'cache_file': cache_file,
                    'cache_size_mb': round(cache_size, 2),
                    'cache_age_hours': round(cache_age, 2),
                    'cache_key': cache_data.get('cache_key', 'unknown'),
                    'num_embeddings': cache_data.get('num_texts', 'unknown'),
                    'cache_version': cache_data.get('version', '1.0')
                }
            except:
                cache_info = {
                    'cache_exists': True,
                    'cache_file': cache_file,
                    'cache_size_mb': round(cache_size, 2),
                    'cache_age_hours': round(cache_age, 2),
                    'error': 'Could not read cache metadata'
                }
        else:
            cache_info = {
                'cache_exists': False,
                'cache_file': cache_file
            }
        
        return {
            'model_name': self.config.model_name,
            'device': self.device,
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.model.max_seq_length,
            'batch_size': self.config.batch_size,
            'cpu_cores': self.config.cpu_cores if self.device == 'cpu' else None,
            'gpu_info': {
                'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 
                            if torch.cuda.is_available() else None
            } if self.device == 'cuda' else None,
            'cache_info': cache_info
        }
    
    def validate_model_health(self) -> Dict[str, Any]:
        """
        Validate model health by running a test encoding.
        
        Returns:
            Dict[str, Any]: Health status dictionary
        """
        try:
            test_text = "This is a test sentence for model validation."
            start_time = time.time()
            
            embedding = self.encode_single(test_text, use_cache=False)
            
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time * 1000, 2),
                'embedding_shape': embedding.shape,
                'embedding_dtype': str(embedding.dtype),
                'device': self.device,
                'timestamp': time.time()
            }
        
        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'device': self.device,
                'timestamp': time.time()
            }