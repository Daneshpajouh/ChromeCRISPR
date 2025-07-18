"""
GeneX Phase 1 - Logging Utilities

Logging utilities for the GeneX mining system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, log_file: Optional[str] = None,
                level: str = 'INFO',
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Decorator to log function calls with parameters.

    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            start_time = datetime.now()
            logger.info(f"Calling {func_name} with args={args}, kwargs={func_kwargs}")

            try:
                result = func(*args, **func_kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"{func_name} completed successfully in {duration:.2f}s")
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.error(f"{func_name} failed after {duration:.2f}s: {e}")
                raise

        return wrapper
    return decorator


def log_performance(logger: logging.Logger, operation: str):
    """
    Decorator to log performance metrics.

    Args:
        logger: Logger instance
        operation: Operation name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"Starting {operation}")

            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"{operation} completed in {duration:.3f}s")
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.error(f"{operation} failed after {duration:.3f}s: {e}")
                raise

        return wrapper
    return decorator


def create_structured_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Create a structured logger with JSON-like formatting.

    Args:
        name: Logger name
        log_file: Path to log file (optional)

    Returns:
        Structured logger instance
    """

    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)

            return str(log_entry)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = StructuredFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_with_context(logger: logging.Logger, context: dict):
    """
    Add context to log messages.

    Args:
        logger: Logger instance
        context: Context dictionary
    """
    class ContextLogger:
        def __init__(self, logger, context):
            self.logger = logger
            self.context = context

        def info(self, message):
            extra_fields = {'context': self.context}
            self.logger.info(message, extra={'extra_fields': extra_fields})

        def error(self, message):
            extra_fields = {'context': self.context}
            self.logger.error(message, extra={'extra_fields': extra_fields})

        def warning(self, message):
            extra_fields = {'context': self.context}
            self.logger.warning(message, extra={'extra_fields': extra_fields})

        def debug(self, message):
            extra_fields = {'context': self.context}
            self.logger.debug(message, extra={'extra_fields': extra_fields})

    return ContextLogger(logger, context)
