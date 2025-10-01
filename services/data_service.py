"""
Data Service - Dataset Loading and Preprocessing
=================================================
This module handles loading FAQ datasets from HuggingFace, preprocessing text,
and caching processed data for performance optimization.
"""

import os
import time
import logging
import threading
from typing import List, Tuple, Optional, Dict, Any
from datasets import load_dataset
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class DataLoader:
    """
    Loads and caches FAQ datasets from HuggingFace.
    
    Handles both small datasets (loaded fully) and large datasets (streaming),
    with intelligent caching to avoid reprocessing on subsequent runs.
    """
    
    def __init__(self, config_service, cache_manager):
        """
        Initialize the DataLoader.
        
        Args:
            config_service: Configuration service instance
            cache_manager: Cache manager for storing processed data
        """
        self.config = config_service.get_model_config()
        self.data_config = config_service.get_data_config()
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        self.dataset = None
        self.df = None
        self.load_lock = threading.RLock()
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load FAQ data from cache or process from dataset.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with questions, answers, and URLs
        """
        with self.load_lock:
            if self._load_from_cache():
                return self.df
            
            return self._process_dataset()
    
    def _load_from_cache(self) -> bool:
        """
        Try to load processed data from cache.
        
        Returns:
            bool: True if cache loaded successfully
        """
        cache_key = f"processed_data_{self.config.dataset_name}_{self.config.max_faqs}"
        cached_data = self.cache_manager.get_file(cache_key)
        
        if cached_data is not None:
            try:
                expected_count = cached_data.get('expected_count', 0)
                max_faqs = self.config.max_faqs or float('inf')
                
                if (len(cached_data['questions']) > 0 and 
                    (not self.config.max_faqs or len(cached_data['questions']) >= min(max_faqs, expected_count * 0.8))):
                    
                    self.df = pd.DataFrame({
                        'question': cached_data['questions'],
                        'answer': cached_data['answers'],
                        'source_url': cached_data['urls']
                    })
                    
                    processing_time_saved = cached_data.get('processing_time', 0)
                    self.logger.info(f"Cached data loaded: {len(self.df)} FAQs")
                    self.logger.info(f"Processing time saved: ~{processing_time_saved:.1f}s")
                    return True
                else:
                    self.logger.info("Cache outdated or incomplete, reprocessing...")
            
            except Exception as e:
                self.logger.warning(f"Cache error: {e}, reprocessing...")
        
        return False
    
    def _process_dataset(self) -> Optional[pd.DataFrame]:
        """
        Process dataset from HuggingFace.
        
        Returns:
            Optional[pd.DataFrame]: Processed FAQ data
        """
        try:
            self.logger.info(f"Loading dataset: {self.config.dataset_name}")
            
            if self.config.max_faqs and self.config.max_faqs < 50000:
                self.logger.info(f"Small dataset mode - loading {self.config.max_faqs} FAQs")
                self.dataset = load_dataset(self.config.dataset_name, streaming=False)
                return self._process_small_dataset()
            else:
                self.logger.info("Large dataset mode - using streaming")
                self.dataset = load_dataset(self.config.dataset_name, streaming=True)
                return self._process_streaming_dataset()
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_small_dataset(self) -> Optional[pd.DataFrame]:
        """
        Process a small dataset fully into memory.
        
        Returns:
            Optional[pd.DataFrame]: Processed data
        """
        start_time = time.time()
        
        train_data = self.dataset['train']
        self.logger.info(f"Dataset loaded. Processing items for {self.config.max_faqs} FAQs...")
        
        items_to_process = []
        questions = []
        answers = []
        urls = []
        
        for i, item in enumerate(train_data):
            if self.config.max_faqs and len(questions) >= self.config.max_faqs:
                break
                
            items_to_process.append(item)
            
            if len(items_to_process) >= 1000 or (self.config.max_faqs and len(questions) + 5000 >= self.config.max_faqs):
                batch_q, batch_a, batch_u = self._process_batch(items_to_process)
                questions.extend(batch_q)
                answers.extend(batch_a)
                urls.extend(batch_u)
                items_to_process = []
                
                if self.config.max_faqs and len(questions) >= self.config.max_faqs:
                    questions = questions[:self.config.max_faqs]
                    answers = answers[:self.config.max_faqs]
                    urls = urls[:self.config.max_faqs]
                    break
                
                self.logger.info(f"Processed {i+1} items, extracted {len(questions)} FAQ pairs")
        
        if items_to_process and (not self.config.max_faqs or len(questions) < self.config.max_faqs):
            batch_q, batch_a, batch_u = self._process_batch(items_to_process)
            questions.extend(batch_q)
            answers.extend(batch_a)
            urls.extend(batch_u)
            
            if self.config.max_faqs and len(questions) > self.config.max_faqs:
                questions = questions[:self.config.max_faqs]
                answers = answers[:self.config.max_faqs]
                urls = urls[:self.config.max_faqs]
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"Data extraction completed:")
        self.logger.info(f"- FAQ pairs extracted: {len(questions)}")
        self.logger.info(f"- Processing time: {processing_time:.1f}s")
        self.logger.info(f"- Speed: {len(questions)/processing_time:.1f} FAQs/second")
        
        self._save_to_cache(questions, answers, urls, processing_time, len(questions))
        
        self.df = pd.DataFrame({
            'question': questions,
            'answer': answers,
            'source_url': urls
        })
        
        return self.df
    
    def _process_batch(self, items_batch: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        """
        Process a batch of items in parallel.
        
        Args:
            items_batch: List of dataset items
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Questions, answers, and URLs
        """
        questions = []
        answers = []
        urls = []
        
        max_workers = min(4, multiprocessing.cpu_count())
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._extract_faqs_from_item, items_batch))
        except:
            self.logger.warning("Parallel processing failed, falling back to sequential")
            results = [self._extract_faqs_from_item(item) for item in items_batch]
        
        for batch_q, batch_a, batch_u in results:
            questions.extend(batch_q)
            answers.extend(batch_a)
            urls.extend(batch_u)
        
        return questions, answers, urls
    
    def _extract_faqs_from_item(self, item: Dict) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract FAQ pairs from a single dataset item.
        
        Args:
            item: Dataset item containing FAQ pairs
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Extracted questions, answers, URLs
        """
        questions = []
        answers = []
        urls = []
        
        faq_pairs = item.get('faq_pairs', [])
        url = item.get('url', '')
        
        for pair in faq_pairs:
            question = pair.get('question', '').strip()
            answer = pair.get('answer', '').strip()
            
            if (len(question) > 5 and len(answer) > 10 and 
                question and answer and
                not question.startswith('http') and  
                len(question) < 1000):
                
                questions.append(question)
                answers.append(answer)
                urls.append(url)
        
        return questions, answers, urls
    
    def _process_streaming_dataset(self) -> Optional[pd.DataFrame]:
        """
        Process a streaming dataset.
        
        Returns:
            Optional[pd.DataFrame]: Processed data
        """
        original_max = self.config.max_faqs
        
        result = self._process_small_dataset()
        
        self.config.max_faqs = original_max
        
        return result
    
    def _save_to_cache(self, questions: List[str], answers: List[str], urls: List[str], 
                       processing_time: float, expected_count: int) -> None:
        """
        Save processed data to cache.
        
        Args:
            questions: List of questions
            answers: List of answers
            urls: List of source URLs
            processing_time: Time taken to process
            expected_count: Expected FAQ count
        """
        try:
            cache_key = f"processed_data_{self.config.dataset_name}_{self.config.max_faqs}"
            cache_data = {
                'questions': questions,
                'answers': answers,
                'urls': urls,
                'processing_time': processing_time,
                'expected_count': expected_count,
                'config_max_faqs': self.config.max_faqs,
                'created_at': time.time(),
                'dataset_name': self.config.dataset_name
            }
            
            success = self.cache_manager.set_file(cache_key, cache_data, compress=True)
            
            if success:
                self.logger.info(f"Data cached successfully")
            else:
                self.logger.warning("Failed to cache processed data")
                
        except Exception as e:
            self.logger.error(f"Cache save failed: {e}")
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: FAQ data
        """
        return self.df
    
    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded data.
        
        Returns:
            Dict[str, Any]: Data statistics
        """
        if self.df is None:
            return {'status': 'no_data_loaded'}
        
        return {
            'total_faqs': len(self.df),
            'avg_question_length': self.df['question'].str.len().mean(),
            'avg_answer_length': self.df['answer'].str.len().mean(),
            'unique_sources': self.df['source_url'].nunique(),
            'dataset_name': self.config.dataset_name,
            'max_faqs_limit': self.config.max_faqs,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }


class TextPreprocessor:
    """
    Preprocesses text by cleaning and normalizing.
    
    Handles lowercasing, punctuation removal, and whitespace normalization.
    """
    
    def __init__(self):
        """Initialize the text preprocessor with regex patterns."""
        import re
        self.punctuation_re = re.compile(r'[\!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]')
        self.digits_re = re.compile(r'\d+')
        self.extra_space_re = re.compile(r'\s+')
        self.logger = logging.getLogger(__name__)
    
    def minimal_clean(self, text: str, lowercase: bool = True, 
                     remove_punctuation: bool = True, remove_digits: bool = True) -> str:
        """
        Apply minimal cleaning to text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_digits: Remove numeric digits
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = self.punctuation_re.sub(' ', text)
        if remove_digits:
            text = self.digits_re.sub('', text)

        text = self.extra_space_re.sub(' ', text)
        text = text.strip()
        
        return text

    def preprocess_questions(self, questions: List[str]) -> List[str]:
        """
        Preprocess a list of questions.
        
        Args:
            questions: List of question strings
            
        Returns:
            List[str]: Preprocessed questions
        """
        return [self.minimal_clean(q, lowercase=True, remove_punctuation=True, 
                                   remove_digits=True) for q in questions]

    def preprocess_user_input(self, user_input: str) -> str:
        """
        Preprocess user input query.
        
        Args:
            user_input: User's question
            
        Returns:
            str: Preprocessed query
        """
        return self.minimal_clean(user_input, lowercase=True, remove_punctuation=True, 
                                 remove_digits=True)


class DataService:
    """
    Main data service coordinating loading and preprocessing.
    
    Combines DataLoader and TextPreprocessor to provide a unified interface
    for accessing FAQ data.
    """
    
    def __init__(self, config_service, cache_manager):
        """
        Initialize the DataService.
        
        Args:
            config_service: Configuration service instance
            cache_manager: Cache manager instance
        """
        self.config_service = config_service
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        self.data_loader = DataLoader(config_service, cache_manager)
        self.preprocessor = TextPreprocessor()
        
        self.questions = None
        self.answers = None
        self.cleaned_questions = None
        
        self.load_lock = threading.RLock()
    
    def load_and_preprocess_data(self) -> bool:
        """
        Load dataset and preprocess all questions.
        
        Returns:
            bool: True if successful
        """
        with self.load_lock:
            try:
                self.logger.info("Loading and preprocessing data...")
                
                df = self.data_loader.load_data()
                if df is None or df.empty:
                    self.logger.error("Failed to load data")
                    return False
                
                self.questions = df['question'].tolist()
                self.answers = df['answer'].tolist()
                
                self.logger.info(f"Loaded {len(self.questions):,} FAQ pairs")
                
                self.logger.info("Preprocessing questions...")
                preprocess_start = time.time()
                
                self.cleaned_questions = self.preprocessor.preprocess_questions(self.questions)
                
                preprocess_time = time.time() - preprocess_start
                self.logger.info(f"Preprocessing completed in {preprocess_time:.2f}s")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error in load_and_preprocess_data: {e}")
                return False
    
    def get_processed_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get processed FAQ data.
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Original questions, answers, 
                                                     and cleaned questions
            
        Raises:
            ValueError: If data not loaded yet
        """
        with self.load_lock:
            if self.questions is None or self.answers is None or self.cleaned_questions is None:
                raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
            
            return self.questions, self.answers, self.cleaned_questions
    
    def preprocess_user_query(self, query: str) -> str:
        """
        Preprocess a user's query.
        
        Args:
            query: User's question
            
        Returns:
            str: Preprocessed query
        """
        return self.preprocessor.preprocess_user_input(query)
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive data information.
        
        Returns:
            Dict[str, Any]: Data statistics and status
        """
        with self.load_lock:
            base_stats = self.data_loader.get_data_stats()
            
            if self.questions is not None:
                base_stats.update({
                    'preprocessing_completed': self.cleaned_questions is not None,
                    'original_questions_count': len(self.questions),
                    'cleaned_questions_count': len(self.cleaned_questions) if self.cleaned_questions else 0
                })
            
            return base_stats