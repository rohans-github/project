"""
Logging Configuration
Sets up logging for the application
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = 'finance_assistant',
    level: str = 'INFO',
    log_to_file: bool = True,
    log_dir: str = 'logs',
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging for the application
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        try:
            # Create log directory
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            
            # Create rotating file handler
            from logging.handlers import RotatingFileHandler
            
            log_file = log_path / f"{name}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            
            # Create error log file
            error_log_file = log_path / f"{name}_errors.log"
            error_handler = RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
