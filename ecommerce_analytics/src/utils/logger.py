import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import get_config


class Logger:
    """Custom logger for the application."""

    def __init__(self, name: str, log_file: Optional[str] = None):
        """Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Path to the log file. If None, use the one from config
        """
        self.config = get_config()
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Get log level from config
        log_level_str = self.config.get('logging.level', 'INFO')
        log_level = getattr(logging, log_level_str)
        self.logger.setLevel(log_level)
        
        # Create formatter
        log_format = self.config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(log_format)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file is None:
            log_file = self.config.get('logging.file', 'logs/app.log')
        
        if log_file:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
            
            # Determine rotation policy
            rotation = self.config.get('logging.rotation', '1 day')
            if 'day' in rotation:
                days = int(rotation.split()[0])
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    log_file, when='D', interval=days, backupCount=7
                )
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10*1024*1024, backupCount=5
                )
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log an exception message."""
        self.logger.exception(message, *args, **kwargs)


def get_logger(name: str) -> Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return Logger(name) 