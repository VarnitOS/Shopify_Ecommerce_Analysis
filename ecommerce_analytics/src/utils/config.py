import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage application configuration from YAML files with environment variable substitution."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, default to "config/config.yaml"
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", 
            "config.yaml"
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from the YAML file with environment variable substitution.
        
        Returns:
            Dict containing the configuration
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config_str = file.read()
        
        # Substitute environment variables
        pattern = r'\${([^:}]+)(?::([^}]+))?}'
        
        def replace_env_var(match):
            env_var, default = match.groups()
            return os.environ.get(env_var, default or '')
        
        config_str = re.sub(pattern, replace_env_var, config_str)
        config = yaml.safe_load(config_str)
        
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "database.url")
            default: Default value if key is not found

        Returns:
            Configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value

    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration.
        
        Returns:
            Dict containing all configuration values
        """
        return self.config


# Create a singleton instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get the configuration singleton.
    
    Returns:
        ConfigLoader instance
    """
    return config 