"""
Cache Service - Multi-layer Caching System
===========================================
This module implements a comprehensive caching system with three layers:
1. LRU Cache: In-memory cache with TTL for fast access
2. File Cache: Persistent disk-based cache for large data
3. Cache Manager: Unified interface coordinating both layers
"""

import os
import pickle
import hashlib
import threading
import time
import json
import logging
import gzip
from typing import Any, Optional, Dict
from collections import OrderedDict
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """
    Represents a single cache entry with metadata.
    
    Attributes:
        data: Cached data object
        created_at: Timestamp of creation
        accessed_at: Timestamp of last access
        access_count: Number of times accessed
        size_bytes: Estimated size in bytes
    """
    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int


class LRUCache:
    """
    Thread-safe Least Recently Used (LRU) cache with TTL.
    
    Automatically evicts least recently used items when capacity is reached.
    Entries expire after a configurable time-to-live period.
    """
    
    def __init__(self, capacity: int = 1000, ttl_hours: int = 24):
        """
        Initialize the LRU cache.
        
        Args:
            capacity: Maximum number of entries
            ttl_hours: Time-to-live in hours for each entry
        """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl_seconds = ttl_hours * 3600
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger(f"{__name__}.LRUCache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached data or None if not found/expired
        """
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if time.time() - entry.created_at > self.ttl_seconds:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Update access metadata
                entry.accessed_at = time.time()
                entry.access_count += 1
                self.cache.move_to_end(key)
                self.hits += 1
                return entry.data
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store an item in cache.
        
        Args:
            key: Cache key
            value: Data to cache
        """
        with self.lock:
            size_bytes = self._estimate_size(value)
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                self.cache[key] = CacheEntry(
                    data=value,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=1,
                    size_bytes=size_bytes
                )
                self.cache.move_to_end(key)
            else:
                # Evict oldest if at capacity
                while len(self.cache) >= self.capacity:
                    oldest_key, _ = self.cache.popitem(last=False)
                    self.logger.debug(f"Evicted cache entry: {oldest_key}")
                
                self.cache[key] = CacheEntry(
                    data=value,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=1,
                    size_bytes=size_bytes
                )
    
    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate the size of an object in bytes.
        
        Args:
            obj: Object to estimate
            
        Returns:
            int: Estimated size in bytes
        """
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default fallback size
    
    def clear_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            int: Number of entries removed
        """
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry.created_at > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Statistics including hit rate, size, etc.
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            avg_access_count = sum(entry.access_count for entry in self.cache.values()) / len(self.cache) if self.cache else 0
            
            return {
                'entries': len(self.cache),
                'capacity': self.capacity,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': round(hit_rate, 2),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'avg_access_count': round(avg_access_count, 1)
            }


class FileCache:
    """
    Persistent file-based cache with compression.
    
    Stores large data structures to disk with optional gzip compression.
    Manages cache size by evicting least accessed files when limit reached.
    """
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 500):
        """
        Initialize the file cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.FileCache")
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        self.metadata = {}
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """
        Generate file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            str: Full file path
        """
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl.gz")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from file cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached data or None if not found
        """
        with self.lock:
            if key not in self.metadata:
                return None
            
            file_path = self._get_file_path(key)
            if not os.path.exists(file_path):
                del self.metadata[key]
                self._save_metadata()
                return None
            
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access metadata
                self.metadata[key]['accessed_at'] = time.time()
                self.metadata[key]['access_count'] += 1
                self._save_metadata()
                
                return data
            
            except Exception as e:
                self.logger.error(f"Failed to load cache file {key}: {e}")
                return None
    
    def set(self, key: str, value: Any, compress: bool = True) -> bool:
        """
        Store data in file cache.
        
        Args:
            key: Cache key
            value: Data to cache
            compress: Whether to use gzip compression
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            file_path = self._get_file_path(key)
            
            try:
                if compress:
                    with gzip.open(file_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                file_size = os.path.getsize(file_path)
                current_time = time.time()
                
                self.metadata[key] = {
                    'created_at': current_time,
                    'accessed_at': current_time,
                    'access_count': 1,
                    'size_bytes': file_size,
                    'compressed': compress
                }
                
                self._cleanup_if_needed()
                self._save_metadata()
                
                self.logger.debug(f"Cached {key} ({file_size / 1024:.1f} KB)")
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to cache {key}: {e}")
                return False
    
    def _cleanup_if_needed(self) -> None:
        """Remove least accessed files if cache size exceeds limit."""
        total_size = sum(entry['size_bytes'] for entry in self.metadata.values())
        
        if total_size > self.max_size_bytes:
            # Sort by access time and count
            entries_by_access = sorted(
                self.metadata.items(),
                key=lambda x: (x[1]['accessed_at'], x[1]['access_count'])
            )
            
            removed_size = 0
            for key, entry in entries_by_access:
                if total_size - removed_size <= self.max_size_bytes * 0.8:
                    break
                
                file_path = self._get_file_path(key)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    removed_size += entry['size_bytes']
                    del self.metadata[key]
                    self.logger.debug(f"Removed cached file: {key}")
                except Exception as e:
                    self.logger.error(f"Failed to remove cache file {key}: {e}")
            
            if removed_size > 0:
                self.logger.info(f"Cache cleanup: removed {removed_size / (1024*1024):.1f} MB")
    
    def clear_expired(self, ttl_hours: int = 24) -> int:
        """
        Remove expired cache files.
        
        Args:
            ttl_hours: Time-to-live in hours
            
        Returns:
            int: Number of files removed
        """
        with self.lock:
            current_time = time.time()
            ttl_seconds = ttl_hours * 3600
            
            expired_keys = [
                key for key, entry in self.metadata.items()
                if current_time - entry['created_at'] > ttl_seconds
            ]
            
            for key in expired_keys:
                file_path = self._get_file_path(key)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    del self.metadata[key]
                except Exception as e:
                    self.logger.error(f"Failed to remove expired cache {key}: {e}")
            
            if expired_keys:
                self._save_metadata()
                self.logger.info(f"Removed {len(expired_keys)} expired cache files")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get file cache statistics.
        
        Returns:
            Dict[str, Any]: Statistics including size and usage
        """
        with self.lock:
            total_size = sum(entry['size_bytes'] for entry in self.metadata.values())
            total_access_count = sum(entry['access_count'] for entry in self.metadata.values())
            avg_access = total_access_count / len(self.metadata) if self.metadata else 0
            
            return {
                'files': len(self.metadata),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
                'usage_percent': round((total_size / self.max_size_bytes) * 100, 1),
                'avg_access_count': round(avg_access, 1)
            }


class CacheManager:
    """
    Unified cache manager coordinating multiple cache layers.
    
    Provides a single interface for:
    - Embedding cache (LRU, in-memory)
    - Query result cache (LRU, in-memory)
    - File cache (persistent, disk-based)
    
    Automatically manages cleanup and expiration across all layers.
    """
    
    def __init__(self, config_service):
        """
        Initialize the cache manager.
        
        Args:
            config_service: Configuration service instance
        """
        self.config = config_service.get_cache_config()
        self.logger = logging.getLogger(__name__)
        
        self.embedding_cache = LRUCache(
            capacity=self.config.embedding_cache_size,
            ttl_hours=self.config.cache_ttl_hours
        )
        
        self.query_cache = LRUCache(
            capacity=self.config.query_cache_size,
            ttl_hours=self.config.cache_ttl_hours
        )
        
        self.file_cache = FileCache()
        
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic cache cleanup."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cache_cleanup_interval_minutes * 60)
                    
                    expired_embedding = self.embedding_cache.clear_expired()
                    expired_query = self.query_cache.clear_expired()
                    expired_files = self.file_cache.clear_expired(self.config.cache_ttl_hours)
                    
                    if expired_embedding + expired_query + expired_files > 0:
                        self.logger.info(f"Cache cleanup completed: {expired_embedding + expired_query + expired_files} items removed")
                
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("Cache cleanup thread started")
    
    def get_embedding(self, key: str) -> Optional[Any]:
        """
        Get an embedding from cache.
        
        Args:
            key: Embedding cache key
            
        Returns:
            Optional[Any]: Cached embedding or None
        """
        return self.embedding_cache.get(key)
    
    def set_embedding(self, key: str, value: Any) -> None:
        """
        Store an embedding in cache.
        
        Args:
            key: Embedding cache key
            value: Embedding data
        """
        self.embedding_cache.set(key, value)
    
    def get_query_result(self, key: str) -> Optional[Any]:
        """
        Get a query result from cache.
        
        Args:
            key: Query cache key
            
        Returns:
            Optional[Any]: Cached result or None
        """
        return self.query_cache.get(key)
    
    def set_query_result(self, key: str, value: Any) -> None:
        """
        Store a query result in cache.
        
        Args:
            key: Query cache key
            value: Query result data
        """
        self.query_cache.set(key, value)
    
    def get_file(self, key: str) -> Optional[Any]:
        """
        Get data from file cache.
        
        Args:
            key: File cache key
            
        Returns:
            Optional[Any]: Cached data or None
        """
        return self.file_cache.get(key)
    
    def set_file(self, key: str, value: Any, compress: bool = True) -> bool:
        """
        Store data in file cache.
        
        Args:
            key: File cache key
            value: Data to cache
            compress: Whether to compress
            
        Returns:
            bool: True if successful
        """
        return self.file_cache.set(key, value, compress)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all cache layers.
        
        Returns:
            Dict[str, Any]: Comprehensive cache statistics
        """
        return {
            'embedding_cache': self.embedding_cache.get_stats(),
            'query_cache': self.query_cache.get_stats(),
            'file_cache': self.file_cache.get_stats(),
            'timestamp': datetime.now().isoformat()
        }