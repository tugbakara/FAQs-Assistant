"""
Search Service - Advanced Semantic Search and Retrieval
=======================================================
This module provides semantic search functionality with multiple indexing strategies,
fuzzy matching, and intelligent fallback mechanisms for robust FAQ retrieval.
"""

import numpy as np
import time
import logging
import threading
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False


class SearchIndex:
    """
    Advanced search index with multiple retrieval strategies.
    
    Supports FAISS-based approximate nearest neighbor search (HNSW, IVF)
    for fast similarity search on large datasets, with fallback to
    brute force cosine similarity for smaller datasets.
    """
    
    def __init__(self, config_service):
        """
        Initialize the search index.
        
        Args:
            config_service: Configuration service instance
        """
        self.config = config_service.get_model_config()
        self.logger = logging.getLogger(__name__)
        
        self.embeddings = None
        self.questions = None
        self.answers = None
        
        self.faiss_index = None
        self.use_faiss = FAISS_AVAILABLE
        
        self.length_buckets = {}
        self.bucket_size = 50
        
        self.index_lock = threading.RLock()
        
        self.index_cache_file = "search_index_cache.pkl"
    
    def fit(self, embeddings: np.ndarray, questions: List[str], answers: List[str]) -> None:
        """
        Build search indexes from embeddings and FAQ data.
        
        Args:
            embeddings: Numpy array of embedding vectors
            questions: List of FAQ questions
            answers: List of FAQ answers
        """
        with self.index_lock:
            self.embeddings = embeddings
            self.questions = questions
            self.answers = answers
            
            if self._load_index_from_cache(embeddings, questions):
                self.logger.info("Loaded search index from cache!")
                return
            
            start_time = time.time()
            self.logger.info(f"Building search indexes for {len(questions):,} items...")
            
            if len(questions) >= 5000 and FAISS_AVAILABLE:
                self._build_optimized_faiss_index()
            else:
                self.logger.info("Using brute force search (optimal for dataset size)")
                self.use_faiss = False
            
            if len(questions) > 50000:
                self._build_length_buckets()
            
            total_time = time.time() - start_time
            self.logger.info(f"Search indexes ready in {total_time:.2f}s!")
            
            self._save_index_to_cache(embeddings, questions)
    
    def _build_length_buckets(self) -> None:
        """
        Build length-based buckets for filtered search.
        
        Groups questions by length to enable length-aware search,
        improving relevance by matching queries to similarly-sized FAQs.
        """
        self.length_buckets = defaultdict(list)
        for i, q in enumerate(self.questions):
            bucket = len(q) // self.bucket_size
            self.length_buckets[bucket].append(i)
        
        self.logger.info(f"Created {len(self.length_buckets)} length buckets")
    
    def _build_optimized_faiss_index(self) -> None:
        """
        Build optimized FAISS index based on dataset size.
        
        Uses FlatL2 for small datasets, simple HNSW for medium datasets,
        and IVF for large datasets to balance speed and accuracy.
        
        Returns:
            bool: True if index built successfully
        """
        try:
            embeddings_np = self.embeddings.astype(np.float32)
            dimension = embeddings_np.shape[1]
            n = len(self.questions)
            
            if n < 10000:
                self.logger.info("Building FlatL2 index (fastest for small datasets)")
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(embeddings_np)
                
            elif n < 100000:
                self.logger.info("Building optimized HNSW index")
                M = 16
                ef_construction = 40
                
                self.faiss_index = faiss.IndexHNSWFlat(dimension, M)
                self.faiss_index.hnsw.ef_construction = ef_construction
                self.faiss_index.add(embeddings_np)
                
            else:
                self.logger.info("Building IVF index for large dataset")
                nlist = max(100, int(np.sqrt(n)))
                quantizer = faiss.IndexFlatL2(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.faiss_index.train(embeddings_np)
                self.faiss_index.add(embeddings_np)
                self.faiss_index.nprobe = 10
            
            self.logger.info(f"FAISS index built: {self.faiss_index.ntotal:,} vectors")
            
        except Exception as e:
            self.logger.warning(f"FAISS build failed: {e}, using brute force")
            self.use_faiss = False
    
    def _save_index_to_cache(self, embeddings: np.ndarray, questions: List[str]) -> None:
        """
        Save FAISS index and metadata to disk cache.
        
        Args:
            embeddings: Embedding vectors
            questions: FAQ questions for validation
        """
        try:
            import pickle
            import hashlib
            
            cache_key = hashlib.md5(
                f"{self.config.model_name}_{len(questions)}_{self.config.dataset_name}".encode()
            ).hexdigest()
            
            cache_data = {
                'cache_key': cache_key,
                'num_questions': len(questions),
                'model_name': self.config.model_name,
                'dataset_name': self.config.dataset_name,
                'use_faiss': self.use_faiss,
                'length_buckets': dict(self.length_buckets),
                'created_at': time.time()
            }
            
            if self.use_faiss and self.faiss_index is not None:
                faiss_file = self.index_cache_file.replace('.pkl', '_faiss.index')
                faiss.write_index(self.faiss_index, faiss_file)
                cache_data['faiss_file'] = faiss_file
                self.logger.info(f"FAISS index saved to {faiss_file}")
            
            with open(self.index_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.info(f"Search index cache saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to save index cache: {e}")
    
    def _load_index_from_cache(self, embeddings: np.ndarray, questions: List[str]) -> bool:
        """
        Load FAISS index from disk cache if valid.
        
        Args:
            embeddings: Current embeddings for validation
            questions: Current questions for validation
            
        Returns:
            bool: True if cache loaded successfully
        """
        try:
            import os
            import pickle
            import hashlib
            
            if not os.path.exists(self.index_cache_file):
                return False
            
            cache_key = hashlib.md5(
                f"{self.config.model_name}_{len(questions)}_{self.config.dataset_name}".encode()
            ).hexdigest()
            
            with open(self.index_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data.get('cache_key') != cache_key:
                self.logger.info("Index cache key mismatch")
                return False
            
            if cache_data.get('num_questions') != len(questions):
                self.logger.info("Index question count mismatch")
                return False
            
            self.length_buckets = defaultdict(list, cache_data.get('length_buckets', {}))
            self.use_faiss = cache_data.get('use_faiss', False)
            
            if self.use_faiss and 'faiss_file' in cache_data:
                faiss_file = cache_data['faiss_file']
                if os.path.exists(faiss_file):
                    self.faiss_index = faiss.read_index(faiss_file)
                    cache_age = (time.time() - cache_data.get('created_at', 0)) / 3600
                    self.logger.info(f"FAISS index loaded from cache (age: {cache_age:.1f}h)")
                else:
                    self.logger.info("FAISS file not found")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load index cache: {e}")
            return False
    
    def clear_index_cache(self) -> None:
        """Clear the search index cache files."""
        try:
            import os
            if os.path.exists(self.index_cache_file):
                os.remove(self.index_cache_file)
            
            faiss_file = self.index_cache_file.replace('.pkl', '_faiss.index')
            if os.path.exists(faiss_file):
                os.remove(faiss_file)
            
            self.logger.info("Search index cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear index cache: {e}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Search for top-k most similar FAQs.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple[List[int], List[float]]: Indices and similarity scores
        """
        with self.index_lock:
            if self.embeddings is None:
                return [], []
            
            k = min(k, len(self.questions))
            
            if self.use_faiss and self.faiss_index is not None:
                return self._faiss_search(query_embedding, k)
            else:
                return self._brute_force_search(query_embedding, k)
    
    def _faiss_search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """
        Search using FAISS index.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            Tuple[List[int], List[float]]: Result indices and similarities
        """
        try:
            query_np = query_embedding.astype(np.float32)
            distances, indices = self.faiss_index.search(query_np, k)
            
            similarities = 1.0 - (distances[0] / 2.0)
            similarities = np.clip(similarities, 0, 1)
            
            valid_mask = indices[0] >= 0
            return indices[0][valid_mask].tolist(), similarities[valid_mask].tolist()
            
        except Exception as e:
            self.logger.error(f"FAISS search error: {e}")
            return self._brute_force_search(query_embedding, k)
    
    def _brute_force_search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """
        Brute force cosine similarity search.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            Tuple[List[int], List[float]]: Result indices and similarities
        """
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        return top_indices.tolist(), top_similarities.tolist()
    
    def length_filtered_search(self, query_text: str, query_embedding: np.ndarray, 
                              k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Search with length-based filtering.
        
        Only searches FAQs with similar length to the query,
        improving relevance for length-sensitive queries.
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding
            k: Number of results
            
        Returns:
            Tuple[List[int], List[float]]: Filtered result indices and similarities
        """
        with self.index_lock:
            input_length = len(query_text)
            tolerance = max(50, input_length // 3)
            
            min_bucket = max(0, (input_length - tolerance) // self.bucket_size)
            max_bucket = (input_length + tolerance) // self.bucket_size
            
            candidate_indices = []
            for bucket_key in range(min_bucket, max_bucket + 1):
                if bucket_key in self.length_buckets:
                    candidate_indices.extend(self.length_buckets[bucket_key])
            
            if not candidate_indices:
                candidate_indices = list(range(min(5000, len(self.questions))))
            else:
                candidate_indices = candidate_indices[:5000]
            
            if len(candidate_indices) < k:
                return candidate_indices, [0.0] * len(candidate_indices)
            
            candidate_embeddings = self.embeddings[candidate_indices]
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            top_indices = np.argsort(similarities)[::-1][:k]
            result_indices = [candidate_indices[i] for i in top_indices]
            result_similarities = [similarities[i] for i in top_indices]
            
            return result_indices, result_similarities
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dict[str, Any]: Index statistics
        """
        with self.index_lock:
            stats = {
                'total_items': len(self.questions) if self.questions else 0,
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'length_buckets': len(self.length_buckets),
                'index_type': 'faiss' if self.use_faiss and self.faiss_index else 'brute_force',
                'faiss_available': FAISS_AVAILABLE
            }
            
            if self.faiss_index:
                stats['index_vectors'] = self.faiss_index.ntotal
            
            return stats


class FuzzyMatcher:
    """
    Fuzzy string matching for finding approximate matches.
    
    Uses fuzzywuzzy library for token-based fuzzy matching,
    useful as a fallback when semantic search fails.
    """
    
    def __init__(self):
        """Initialize the fuzzy matcher."""
        self.logger = logging.getLogger(__name__)
        self.available = FUZZYWUZZY_AVAILABLE
        
        if not self.available:
            self.logger.warning("fuzzywuzzy not available, fuzzy matching disabled")
    
    def find_fuzzy_matches(self, query: str, questions: List[str], 
                          min_score: int = 70, max_candidates: int = 1000) -> List[Tuple[str, int, int]]:
        """
        Find fuzzy matches for a query.
        
        Args:
            query: User's query
            questions: List of FAQ questions
            min_score: Minimum match score (0-100)
            max_candidates: Maximum questions to check
            
        Returns:
            List[Tuple[str, int, int]]: Matched questions with scores and indices
        """
        if not self.available or not query or not questions:
            return []
        
        input_length = len(query)
        length_tolerance = max(30, input_length // 2)
        
        filtered_questions = []
        for i, q in enumerate(questions):
            if abs(len(q) - input_length) <= length_tolerance:
                filtered_questions.append((q, i))
            if len(filtered_questions) >= max_candidates:
                break
        
        if not filtered_questions:
            return []
        
        matches = []
        for question, original_idx in filtered_questions:
            score = fuzz.token_sort_ratio(query, question)
            if score >= min_score:
                matches.append((question, score, original_idx))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]


class SearchService:
    """
    Main search service coordinating all search strategies.
    
    Combines semantic search, fuzzy matching, and spell correction
    for robust FAQ retrieval with multiple fallback mechanisms.
    """
    
    def __init__(self, config_service, cache_manager, embedding_service, spell_correction_service=None):
        """
        Initialize the search service.
        
        Args:
            config_service: Configuration service
            cache_manager: Cache manager
            embedding_service: Embedding generation service
            spell_correction_service: Optional spell correction service
        """
        self.config = config_service.get_model_config()
        self.cache_manager = cache_manager
        self.embedding_service = embedding_service
        self.spell_correction_service = spell_correction_service
        self.logger = logging.getLogger(__name__)
        
        self.search_index = SearchIndex(config_service)
        self.fuzzy_matcher = FuzzyMatcher()
        
        if self.spell_correction_service and self.spell_correction_service.is_available:
            self.logger.info("Spell correction service integrated")
        else:
            self.logger.info("Spell correction service not available")
        
        self.is_fitted = False
        self.fit_lock = threading.RLock()
    
    def fit(self, embeddings: np.ndarray, questions: List[str], answers: List[str]) -> None:
        """
        Fit the search service with FAQ data.
        
        Args:
            embeddings: FAQ embeddings
            questions: FAQ questions
            answers: FAQ answers
        """
        with self.fit_lock:
            self.search_index.fit(embeddings, questions, answers)
            self.is_fitted = True
            self.logger.info("Search service ready!")
    
    def find_best_match(self, user_question: str) -> Tuple[Optional[str], float]:
        """
        Find the best matching FAQ for a user's question.
        
        Uses semantic search with intelligent fallbacks:
        1. Semantic similarity search
        2. Spell correction + retry
        3. Fuzzy string matching
        
        Args:
            user_question: User's question
            
        Returns:
            Tuple[Optional[str], float]: Best answer and similarity score
        """
        if not self.is_fitted:
            return None, 0.0
        
        query_hash = self._get_query_hash(user_question)
        cached_result = self.cache_manager.get_query_result(query_hash)
        if cached_result:
            self.logger.debug(f"Cache hit for query: '{user_question}'")
            return cached_result
        
        self.logger.info(f"Query: '{user_question}' (length: {len(user_question)})")
        start_time = time.time()
        
        query_embedding = self.embedding_service.encode_single(user_question, normalize=True)
        query_embedding = query_embedding.reshape(1, -1)
        
        best_indices, best_similarities = self.search_index.search(query_embedding, k=5)
        
        search_time = time.time() - start_time
        
        if best_similarities and best_similarities[0] >= self.config.similarity_threshold:
            best_idx = best_indices[0]
            best_similarity = best_similarities[0]
            result = (self.search_index.answers[best_idx], best_similarity)
            
            self.cache_manager.set_query_result(query_hash, result)
            
            confidence_level = "High" if best_similarity > 0.85 else "Good"
            self.logger.info(f"Match found: {best_similarity:.4f} ({confidence_level}) | Time: {search_time*1000:.1f}ms")
            return result
        
        fallback_result = self._fallback_search(user_question, query_hash, search_time)
        if fallback_result:
            return fallback_result
        
        total_time = time.time() - start_time
        self.logger.info(f"No match found | Time: {total_time*1000:.1f}ms")
        
        no_match_result = (None, best_similarities[0] if best_similarities else 0.0)
        self.cache_manager.set_query_result(query_hash, no_match_result)
        return no_match_result
    
    def _fallback_search(self, user_question: str, query_hash: str, 
                        initial_search_time: float) -> Optional[Tuple[str, float]]:
        """
        Execute fallback search strategies.
        
        Args:
            user_question: User's question
            query_hash: Query cache key
            initial_search_time: Time spent on initial search
            
        Returns:
            Optional[Tuple[str, float]]: Result from fallback or None
        """
        fallback_start = time.time()
        
        if (self.spell_correction_service and 
            self.spell_correction_service.is_available and 
            len(user_question) < 200):
            
            self.logger.info("Trying spell correction...")
            
            try:
                correction_result = self.spell_correction_service.correct_text(user_question)
                
                if correction_result['has_corrections']:
                    corrected_query = correction_result['corrected_text']
                    self.logger.info(f"Corrected: '{corrected_query}'")
                    
                    corrected_result = self.find_best_match(corrected_query)
                    if corrected_result[0] and corrected_result[1] >= self.config.similarity_threshold * 0.85:
                        self.cache_manager.set_query_result(query_hash, corrected_result)
                        
                        fallback_time = time.time() - fallback_start
                        self.logger.info(f"Spell correction match | Time: {(initial_search_time + fallback_time)*1000:.1f}ms")
                        return corrected_result
            except Exception as e:
                self.logger.error(f"Spell correction error: {e}")
        
        if FUZZYWUZZY_AVAILABLE and len(user_question) < 300:
            self.logger.info("Trying fuzzy matching...")
            
            matches = self.fuzzy_matcher.find_fuzzy_matches(
                user_question, 
                self.search_index.questions,
                min_score=70,
                max_candidates=500
            )
            
            if matches:
                best_fuzzy = matches[0]
                fuzzy_similarity = best_fuzzy[1] / 100.0
                result = (self.search_index.answers[best_fuzzy[2]], fuzzy_similarity)
                
                self.cache_manager.set_query_result(query_hash, result)
                
                fallback_time = time.time() - fallback_start
                self.logger.info(f"Fuzzy match: {best_fuzzy[1]}/100 | Time: {(initial_search_time + fallback_time)*1000:.1f}ms")
                return result
        
        return None
    
    def get_top_matches(self, user_question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top-k matching FAQs.
        
        Args:
            user_question: User's question
            top_k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Top matches with metadata
        """
        if not self.is_fitted:
            return []
        
        query_embedding = self.embedding_service.encode_single(user_question, normalize=True)
        query_embedding = query_embedding.reshape(1, -1)
        
        best_indices, best_similarities = self.search_index.search(query_embedding, k=top_k)
        
        results = []
        for i, (idx, sim) in enumerate(zip(best_indices, best_similarities)):
            results.append({
                'rank': i + 1,
                'question': self.search_index.questions[idx],
                'answer': self.search_index.answers[idx],
                'similarity': sim,
                'confidence': 'High' if sim > 0.85 else 'Medium' if sim > 0.7 else 'Low'
            })
        
        return results
    
    def _get_query_hash(self, query: str) -> str:
        """
        Generate cache key for a query.
        
        Args:
            query: Query string
            
        Returns:
            str: MD5 hash
        """
        import hashlib
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search service statistics.
        
        Returns:
            Dict[str, Any]: Search statistics
        """
        return {
            'is_fitted': self.is_fitted,
            'index_stats': self.search_index.get_index_stats() if self.is_fitted else {},
            'fuzzy_available': self.fuzzy_matcher.available,
            'cache_stats': self.cache_manager.get_comprehensive_stats()
        }