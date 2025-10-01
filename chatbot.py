"""
FAQ Chatbot Engine - Core Business Logic
=========================================
This module implements the main chatbot engine that orchestrates various services
including caching, search, and response generation for FAQ queries.
"""

import logging
import time
from typing import Dict, Any, List
from services.config_service import config_service
from services.cache_service import CacheManager
from services.orchestrator_service import OrchestratorService


class ChatbotEngine:
    """
    Main chatbot engine class that handles FAQ query processing.
    
    This class serves as the primary interface for the chatbot functionality,
    coordinating between various microservices to provide accurate FAQ responses.
    
    Attributes:
        logger (logging.Logger): Logger instance for this class
        cache_manager (CacheManager): Manages caching for embeddings and queries
        orchestrator (OrchestratorService): Orchestrates various chatbot services
    """
    
    def __init__(self):
        """
        Initialize the ChatbotEngine with required services.
        
        Sets up logging, cache management, and the orchestrator service
        for handling FAQ queries.
        """
        self.logger = logging.getLogger(__name__)
        
        self.cache_manager = CacheManager(config_service)
        self.orchestrator = OrchestratorService(config_service, self.cache_manager)
        
        self.logger.info("ChatbotEngine initialized with microservices architecture")
    
    def initialize(self) -> bool:
        """
        Initialize the chatbot system.
        
        Starts the initialization process for loading models, embeddings,
        and other required resources.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.orchestrator.start_initialization()
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def get_response(self, user_input: str) -> str:
        """
        Get a response for a user's FAQ query.
        
        Args:
            user_input (str): The user's question or query
            
        Returns:
            str: The chatbot's response containing the answer or relevant FAQ
        """
        return self.orchestrator.get_response(user_input)
    
    def get_multiple_responses(self, user_input: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get multiple top matching responses for a query.
        
        Useful for providing alternative answers or showing confidence levels
        across multiple potential matches.
        
        Args:
            user_input (str): The user's question or query
            top_k (int, optional): Number of top matches to return. Defaults to 3.
            
        Returns:
            List[Dict[str, Any]]: List of top matching FAQ entries with metadata,
                                  empty list if chatbot not initialized or error occurs
        """
        if not self.orchestrator.is_initialized:
            return []
        
        try:
            return self.orchestrator.search_service.get_top_matches(user_input, top_k)
        except Exception as e:
            self.logger.error(f"Error getting multiple responses: {e}")
            return []
    
    def test_queries(self, test_queries: List[str]) -> None:
        """
        Test the chatbot with a list of queries and log performance metrics.
        
        Runs through a list of test queries and reports success rate,
        average response time, and other system statistics.
        
        Args:
            test_queries (List[str]): List of test questions to evaluate
        """
        if not self.orchestrator.is_initialized:
            self.logger.error("Chatbot not initialized!")
            return
        
        successful_matches = 0
        total_response_time = 0
        
        for i, query in enumerate(test_queries, 1):
            self.logger.info(f"--- Test {i}/{len(test_queries)} ---")
            start_time = time.time()
            response = self.get_response(query)
            response_time = time.time() - start_time
            total_response_time += response_time
            
            if response and "No relevant answer found" not in response:
                successful_matches += 1
            
            self.logger.info(f"Response: {response[:150]}...")
        
        success_rate = (successful_matches / len(test_queries)) * 100
        avg_response_time = total_response_time / len(test_queries)
        
        stats = self.orchestrator.get_system_stats()
        
        self.logger.info("=" * 70)
        self.logger.info("MICROSERVICES TEST RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Success: {success_rate:.1f}% ({successful_matches}/{len(test_queries)})")
        self.logger.info(f"Avg time: {avg_response_time*1000:.1f}ms")
        
        if 'data_stats' in stats:
            self.logger.info(f"Dataset: {stats['data_stats'].get('total_faqs', 0):,} FAQs")
        
        self.logger.info("=" * 70)
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """
        Get the current initialization status of the chatbot.
        
        Returns:
            Dict[str, Any]: Dictionary containing initialization status including:
                           - started: bool
                           - completed: bool
                           - error: str or None
                           - progress: str
                           - stage: int
                           - total_stages: int
        """
        return self.orchestrator.get_status()
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Perform a health check on all chatbot subsystems.
        
        Returns:
            Dict[str, Any]: Dictionary containing health status of all services
        """
        return self.orchestrator.health_check()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the chatbot system.
        
        Returns:
            Dict[str, Any]: Dictionary containing detailed system statistics including:
                           - data_stats: Dataset information
                           - cache_stats: Cache performance metrics
                           - model_stats: Model information
                           - performance_stats: Response times and success rates
        """
        return self.orchestrator.get_system_stats()