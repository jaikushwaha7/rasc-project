"""
Configuration management for RASC project
Loads and validates configuration from YAML files
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration loader and accessor"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            # Default to configs/config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'detection.training.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._deep_update(self._config, updates)
    
    @staticmethod
    def _deep_update(base: dict, updates: dict):
        """Recursively update nested dictionaries"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                Config._deep_update(base[key], value)
            else:
                base[key] = value


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object
    """
    return Config(config_path)


# Singleton instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton)
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config object
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config
