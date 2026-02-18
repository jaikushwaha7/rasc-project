"""
Logging utilities for RASC project
Provides structured logging with file and console handlers
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """Centralized logging manager for RASC"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: Optional[str] = "logs",
        level: str = "INFO",
        console: bool = True,
        file_logging: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger instance
        
        Args:
            name: Logger name (usually __name__)
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console: Enable console output
            file_logging: Enable file output
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if file_logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # Main log file
            log_file = os.path.join(
                log_dir,
                f"{name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


class ExperimentLogger:
    """Enhanced logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger.get_logger(
            f"experiment.{experiment_name}",
            log_dir=str(self.log_dir)
        )
        
        self.metrics = {}
        self.config = {}
        
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.config = config
        self.logger.info(f"Experiment configuration: {config}")
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
        
        if step is not None:
            self.logger.info(f"{name} at step {step}: {value:.4f}")
        else:
            self.logger.info(f"{name}: {value:.4f}")
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an entire epoch"""
        msg = f"Epoch {epoch}: " + ", ".join(
            [f"{k}={v:.4f}" for k, v in metrics.items()]
        )
        self.logger.info(msg)
        
        for name, value in metrics.items():
            self.log_metric(name, value, step=epoch)
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        import json
        metrics_file = self.log_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")


# Convenience function
def get_logger(name: str, **kwargs) -> logging.Logger:
    """Get a logger instance (wrapper for Logger.get_logger)"""
    return Logger.get_logger(name, **kwargs)
