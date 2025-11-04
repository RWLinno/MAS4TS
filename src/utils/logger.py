#!/usr/bin/env python3
"""
Logging Utilities
Centralized logging configuration for OnCallAgent
"""

import logging
import logging.handlers
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logger(log_level: str = "INFO", config: Dict[str, Any] = None) -> logging.Logger:
    """
    Setup centralized logging for MAS4TS
    
    Args:
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
        config: Logging configuration dictionary (optional)
        
    Returns:
        Configured logger instance
    """
    config = config or {}
    
    # Get configuration values
    if isinstance(log_level, str):
        log_level = log_level.upper()
    else:
        log_level = config.get("level", "INFO").upper()
    
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_dir = config.get("log_dir", "logs")
    max_file_size = config.get("max_file_size_mb", 10) * 1024 * 1024  # Convert to bytes
    backup_count = config.get("backup_count", 5)
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "mas4ts.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # File gets all messages
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "mas4ts_errors.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log handler (optional)
    if config.get("enable_performance_logging", False):
        perf_handler = logging.handlers.RotatingFileHandler(
            log_path / "oncall_agent_performance.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(logging.Formatter(
            "%(asctime)s - PERF - %(message)s"
        ))
        
        # Create performance logger
        perf_logger = logging.getLogger("oncall_agent.performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
    
    logger = logging.getLogger("mas4ts")
    logger.info("✓ Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_path.absolute()}")
    
    return logger

def get_performance_logger() -> logging.Logger:
    """Get performance logger for metrics tracking"""
    return logging.getLogger("oncall_agent.performance")

def log_performance_metric(metric_name: str, value: Any, context: Dict[str, Any] = None):
    """Log performance metric"""
    perf_logger = get_performance_logger()
    
    metric_data = {
        "metric": metric_name,
        "value": value,
        "context": context or {}
    }
    
    perf_logger.info(f"METRIC: {json.dumps(metric_data)}")

class StructuredLogger:
    """
    Structured logging for better observability
    Useful for production deployments and debugging
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_agent_execution(self, agent_name: str, query: str, 
                          result: Dict[str, Any], processing_time: float):
        """Log agent execution with structured data"""
        log_data = {
            "event": "agent_execution",
            "agent": agent_name,
            "query_length": len(query),
            "confidence": result.get("confidence", 0.0),
            "processing_time": processing_time,
            "success": result.get("success", False)
        }
        
        self.logger.info(f"AGENT_EXEC: {json.dumps(log_data)}")
    
    def log_routing_decision(self, query: str, selected_agent: str, 
                           confidence: float, reasoning: str):
        """Log routing decisions for analysis"""
        log_data = {
            "event": "routing_decision",
            "query_length": len(query),
            "selected_agent": selected_agent,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
        self.logger.info(f"ROUTING: {json.dumps(log_data)}")
    
    def log_error(self, component: str, error: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        log_data = {
            "event": "error",
            "component": component,
            "error": error,
            "context": context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(log_data)}")
    
    def log_performance(self, operation: str, duration: float, 
                       metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        log_data = {
            "event": "performance",
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }
        
        self.logger.info(f"PERF: {json.dumps(log_data)}")
