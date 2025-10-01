"""
Configuration Service - Centralized Configuration Management
=============================================================
This module manages all application configuration with dataclasses
and provides easy access to configuration sections.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging


@dataclass
class ModelConfig:
    """Model and embedding configuration."""
    dataset_name: str = "vishal-burman/c4-faqs"
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    similarity_threshold: float = 0.78
    max_faqs: Optional[int] = None
    cache_file: str = "faq_embeddings.pkl"
    embeddings_cache_file: str = "faq_embeddings_cache.pkl"
    batch_size: int = 64
    use_gpu: bool = True
    cpu_cores: int = 4
    embedding_threads: int = 4
    force_regenerate_embeddings: bool = False


@dataclass
class UIConfig:
    """User interface configuration."""
    app_title: str = "FAQ Chatbot"
    app_host: str = "127.0.0.1"
    app_port: int = 8050
    debug_mode: bool = False
    font_playfair_path: str = "fonts/Playfair_Display/static/PlayfairDisplay-SemiBold.ttf"
    font_century_path: str = "fonts/CenturyGothic/centurygothic.ttf"
    icon_enter_path: str = "icons/enter.png"
    icon_ai_path: str = "icons/ai.png"
    icon_user_path: str = "icons/user.png"
    background_images_folder: str = "bg"


@dataclass
class CacheConfig:
    """Caching configuration."""
    embedding_cache_size: int = 1000
    query_cache_size: int = 5000
    cache_ttl_hours: int = 24
    cache_cleanup_interval_minutes: int = 60


@dataclass
class DataConfig:
    """Data processing configuration."""
    processed_cache_file: str = "processed_faqs.pkl"
    backup_cache_interval_hours: int = 6
    max_cache_file_size_mb: int = 500


class ConfigService:
    """
    Centralized configuration service.
    
    Loads configuration from JSON file and provides typed access
    to configuration sections.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration service.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        self.model_config = ModelConfig()
        self.ui_config = UIConfig()
        self.cache_config = CacheConfig()
        self.data_config = DataConfig()
        
        self._load_config()
        self._validate_config()
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                if 'model' in config_data:
                    self.model_config = ModelConfig(**config_data['model'])
                if 'ui' in config_data:
                    self.ui_config = UIConfig(**config_data['ui'])
                if 'cache' in config_data:
                    self.cache_config = CacheConfig(**config_data['cache'])
                if 'data' in config_data:
                    self.data_config = DataConfig(**config_data['data'])
                    
                self.logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
                self._save_default_config()
        else:
            self.logger.info("Config file not found, creating default configuration")
            self._save_default_config()
    
    def _save_default_config(self) -> None:
        """Save default configuration to file."""
        config_data = {
            'model': asdict(self.model_config),
            'ui': asdict(self.ui_config),
            'cache': asdict(self.cache_config),
            'data': asdict(self.data_config)
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            self.logger.info(f"Default configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration values and directories."""
        # Validate font directory
        if not os.path.exists(os.path.dirname(self.ui_config.font_playfair_path) 
                            if os.path.dirname(self.ui_config.font_playfair_path) else "."):
            self.logger.warning(f"Font directory not found: {os.path.dirname(self.ui_config.font_playfair_path)}")
        
        # Validate icon directory
        if not os.path.exists(os.path.dirname(self.ui_config.icon_enter_path) 
                            if os.path.dirname(self.ui_config.icon_enter_path) else "."):
            self.logger.warning(f"Icon directory not found: {os.path.dirname(self.ui_config.icon_enter_path)}")
        
        # Validate batch size
        if self.model_config.batch_size <= 0:
            self.logger.warning("Invalid batch_size, setting to 32")
            self.model_config.batch_size = 32
        
        # Validate CPU cores
        if self.model_config.cpu_cores <= 0:
            import multiprocessing
            self.model_config.cpu_cores = multiprocessing.cpu_count()
            self.logger.info(f"CPU cores set to {self.model_config.cpu_cores}")
        
        # Log cache file configuration
        self.logger.info(f"Embeddings cache file: {self.model_config.embeddings_cache_file}")
        if os.path.exists(self.model_config.embeddings_cache_file):
            cache_size = os.path.getsize(self.model_config.embeddings_cache_file) / (1024 * 1024)
            self.logger.info(f"Existing embeddings cache found: {cache_size:.1f} MB")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model_config
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self.ui_config
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        return self.cache_config
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.data_config
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration values dynamically.
        
        Args:
            section: Configuration section name ('model', 'ui', 'cache', 'data')
            updates: Dictionary of key-value pairs to update
        """
        if section == 'model':
            for key, value in updates.items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        elif section == 'ui':
            for key, value in updates.items():
                if hasattr(self.ui_config, key):
                    setattr(self.ui_config, key, value)
        elif section == 'cache':
            for key, value in updates.items():
                if hasattr(self.cache_config, key):
                    setattr(self.cache_config, key, value)
        elif section == 'data':
            for key, value in updates.items():
                if hasattr(self.data_config, key):
                    setattr(self.data_config, key, value)
        
        self._save_default_config()
        self.logger.info(f"Configuration updated: {section}")


# Global configuration service instance
config_service = ConfigService()

# Legacy constants for backwards compatibility
DATASET_NAME = config_service.model_config.dataset_name
MODEL_NAME = config_service.model_config.model_name
SIMILARITY_THRESHOLD = config_service.model_config.similarity_threshold
MAX_FAQS = config_service.model_config.max_faqs
CACHE_FILE = config_service.model_config.cache_file
EMBEDDINGS_CACHE_FILE = config_service.model_config.embeddings_cache_file
BATCH_SIZE = config_service.model_config.batch_size
USE_GPU = config_service.model_config.use_gpu
CPU_CORES = config_service.model_config.cpu_cores
EMBEDDING_THREADS = config_service.model_config.embedding_threads
FORCE_REGENERATE_EMBEDDINGS = config_service.model_config.force_regenerate_embeddings

APP_TITLE = config_service.ui_config.app_title
APP_HOST = config_service.ui_config.app_host
APP_PORT = config_service.ui_config.app_port
DEBUG_MODE = config_service.ui_config.debug_mode
FONT_PLAYFAIR_PATH = config_service.ui_config.font_playfair_path
FONT_CENTURY_PATH = config_service.ui_config.font_century_path
ICON_ENTER_PATH = config_service.ui_config.icon_enter_path
ICON_AI_PATH = config_service.ui_config.icon_ai_path
ICON_USER_PATH = config_service.ui_config.icon_user_path
BACKGROUND_IMAGES_FOLDER = config_service.ui_config.background_images_folder