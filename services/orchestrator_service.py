"""
Orchestrator Service - Central Coordination of All Services
===========================================================
This module orchestrates initialization and coordination of all chatbot services
including data loading, embedding generation, and search functionality.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class InitializationStatus:
    """
    Tracks the initialization status of the chatbot system.
    
    Attributes:
        started: Whether initialization has begun
        completed: Whether initialization finished successfully
        error: Error message if initialization failed
        progress: Current progress description
        stage: Current initialization stage name
        total_stages: Total number of initialization stages
    """
    started: bool = False
    completed: bool = False
    error: Optional[str] = None
    progress: str = "Waiting..."
    stage: str = "init"
    total_stages: int = 4


class OrchestratorService:
    """
    Main orchestrator coordinating all chatbot services.
    
    This service manages the initialization sequence, coordinates between
    different services, and handles query processing through the system.
    """
    
    def __init__(self, config_service, cache_manager):
        """
        Initialize the Orchestrator Service.
        
        Args:
            config_service: Configuration service instance
            cache_manager: Cache manager instance
        """
        self.config_service = config_service
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        self.embedding_service = None
        self.search_service = None
        self.data_service = None
        self.spell_correction_service = None
        
        self.initialization_status = InitializationStatus()
        self.status_lock = threading.RLock()
        
        self.is_initialized = False
        self.init_lock = threading.RLock()
    
    def start_initialization(self) -> None:
        """
        Start the initialization process in a background thread.
        
        This allows the web application to remain responsive while
        the system initializes.
        """
        with self.status_lock:
            if self.initialization_status.started:
                self.logger.warning("Initialization already started")
                return
            
            self.initialization_status.started = True
            self.initialization_status.progress = "Starting initialization..."
        
        init_thread = threading.Thread(target=self._background_initialization, daemon=True)
        init_thread.start()
        self.logger.info("Background initialization started")
    
    def _background_initialization(self) -> None:
        """
        Execute the complete initialization sequence in the background.
        
        Steps:
        1. Initialize all services
        2. Load and preprocess data
        3. Generate embeddings (with caching)
        4. Setup search indexes
        """
        try:
            self._update_status(stage="services", progress="Initializing services...")
            self._initialize_services()
            
            self._update_status(stage="data", progress="Loading and preprocessing data...")
            if not self._load_data():
                raise Exception("Data loading failed")
            
            self._update_status(stage="embeddings", progress="Generating embeddings...")
            if not self._generate_embeddings():
                raise Exception("Embedding generation failed")
            
            self._update_status(stage="search", progress="Building search indexes...")
            if not self._setup_search():
                raise Exception("Search setup failed")
            
            with self.status_lock:
                self.initialization_status.completed = True
                self.initialization_status.progress = "Ready!"
                self.initialization_status.stage = "complete"
            
            with self.init_lock:
                self.is_initialized = True
            
            self.logger.info("System initialization completed successfully!")
            
        except Exception as e:
            error_msg = str(e)
            with self.status_lock:
                self.initialization_status.error = error_msg
                self.initialization_status.progress = f"Error: {error_msg}"
            
            self.logger.error(f"System initialization failed: {e}")
    
    def _initialize_services(self) -> None:
        """
        Initialize all required services.
        
        Services initialized:
        - EmbeddingService: For generating text embeddings
        - DataService: For loading and preprocessing FAQs
        - SpellCorrectionService: For correcting user queries
        - SearchService: For finding relevant FAQs
        """
        from .embedding_service import EmbeddingService
        from .search_service import SearchService
        from .data_service import DataService
        from .spell_correction_service import SpellCorrectionService
        
        self.embedding_service = EmbeddingService(self.config_service, self.cache_manager)
        self.data_service = DataService(self.config_service, self.cache_manager)
        self.spell_correction_service = SpellCorrectionService(self.config_service, self.cache_manager)
        self.search_service = SearchService(
            self.config_service, 
            self.cache_manager, 
            self.embedding_service,
            self.spell_correction_service
        )
        
        self.logger.info("All services initialized")
    
    def _load_data(self) -> bool:
        """
        Load and preprocess FAQ data.
        
        Returns:
            bool: True if data loaded successfully
        """
        return self.data_service.load_and_preprocess_data()
    
    def _generate_embeddings(self) -> bool:
        """
        Generate embeddings for all FAQ questions.
        
        Uses persistent caching to avoid regeneration on subsequent runs.
        
        Returns:
            bool: True if embeddings generated/loaded successfully
        """
        try:
            questions, answers, cleaned_questions = self.data_service.get_processed_data()
            
            # Train spell correction service with FAQ data
            if (self.spell_correction_service and 
                self.spell_correction_service.is_available):
                self.logger.info("Training spell correction with FAQ data...")
                sample_text = " ".join(questions[:1000])
                self.spell_correction_service.learn_from_text(sample_text)
            
            self.logger.info(f"Generating embeddings for {len(cleaned_questions):,} questions...")
            
            # This will use cache if available
            embeddings = self.embedding_service.encode_batch(
                cleaned_questions, 
                normalize=True, 
                use_cache=True
            )
            
            self.embeddings = embeddings
            self.questions = questions
            self.answers = answers
            self.cleaned_questions = cleaned_questions
            
            self.logger.info(f"Embeddings ready: {embeddings.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return False
    
    def _setup_search(self) -> bool:
        """
        Setup search indexes for fast retrieval.
        
        Returns:
            bool: True if search setup successful
        """
        try:
            self.search_service.fit(self.embeddings, self.questions, self.answers)
            return True
        except Exception as e:
            self.logger.error(f"Search setup failed: {e}")
            return False
    
    def _update_status(self, stage: str, progress: str) -> None:
        """
        Update initialization status.
        
        Args:
            stage: Current stage name
            progress: Progress description
        """
        with self.status_lock:
            self.initialization_status.stage = stage
            self.initialization_status.progress = progress
        self.logger.info(f"[{stage}] {progress}")
    
    def get_response(self, user_input: str) -> str:
        """
        Get a response for a user's question.
        
        Args:
            user_input: User's question
            
        Returns:
            str: Answer from FAQ database or error message
        """
        if not self.is_initialized:
            if self.initialization_status.error:
                return "System initialization failed. Please restart the application."
            else:
                return "System is still initializing. Please wait..."
        
        if not user_input or len(user_input.strip()) < 3:
            return "Please ask a more specific question."
        
        try:
            answer, similarity = self.search_service.find_best_match(user_input.strip())
            
            if answer:
                confidence = "High" if similarity > 0.8 else "Medium" if similarity > 0.7 else "Good"
                self.logger.info(f"Match confidence: {confidence}")
                return answer
            else:
                return "No relevant answer found from FAQs. Try rephrasing your question or check your spelling."
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return "Sorry, there was an error processing your question. Please try again."
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current initialization status.
        
        Returns:
            Dict[str, Any]: Status information including progress and errors
        """
        with self.status_lock:
            return {
                "started": self.initialization_status.started,
                "completed": self.initialization_status.completed,
                "error": self.initialization_status.error,
                "progress": self.initialization_status.progress,
                "stage": self.initialization_status.stage,
                "total_stages": self.initialization_status.total_stages,
                "is_initialized": self.is_initialized
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict[str, Any]: Statistics from all services
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        try:
            stats = {
                "system_status": "ready",
                "initialization_time": getattr(self, 'init_time', None),
                "data_stats": self.data_service.get_data_info(),
                "embedding_stats": self.embedding_service.get_model_info(),
                "search_stats": self.search_service.get_search_stats(),
                "cache_stats": self.cache_manager.get_comprehensive_stats()
            }
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all system components.
        
        Returns:
            Dict[str, Any]: Health status of each component
        """
        try:
            health_status = {
                "overall": "healthy",
                "timestamp": time.time(),
                "components": {}
            }
            
            if self.is_initialized:
                health_status["components"]["embedding_service"] = self.embedding_service.validate_model_health()
                health_status["components"]["search_service"] = {"status": "healthy"}
                health_status["components"]["data_service"] = {"status": "healthy"}
                health_status["components"]["cache_manager"] = {"status": "healthy"}
            else:
                health_status["overall"] = "initializing"
                health_status["initialization_status"] = self.get_status()
            
            return health_status
        
        except Exception as e:
            return {
                "overall": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }