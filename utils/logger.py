#!/usr/bin/env python3
"""
Logging utilities for Multi-Robot Coordination Framework
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(f"perf_{name}", f"logs/performance_{name}.log")
    
    def log_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Log performance metric"""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        self.logger.info(f"METRIC:{metric_name}:{value}:{timestamp}")
