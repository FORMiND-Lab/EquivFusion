import os
import logging
from pathlib import Path

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.ERROR:
            # 为 ERROR 及以上级别添加前缀
            level_name = logging.getLevelName(record.levelno)
            return f"[{level_name}] {super().format(record)}"
        return super().format(record)

class LogUtil:
    @staticmethod
    def init_logger(log_dir):
        """
        Configure logging to write identical messages to both a file and console.

        Args:
            log_dir: Directory path (str or Path) where log file will be stored

        Returns:
            Configured root logger instance
        """
        # Standardize path handling using Path object
        log_file = Path(log_dir) / "result.log"

        # Create formatter
        formatter = CustomFormatter('%(message)s')

        # Get root logger and clear existing handlers to prevent duplicate logging
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()

        # Set global logging level to INFO
        logger.setLevel(logging.INFO)

        # Configure file handler to write to log file
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Configure console handler to output to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def clean_logger(logger):
        # Clean up log handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

