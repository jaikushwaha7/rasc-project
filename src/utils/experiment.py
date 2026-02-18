"""
Experiment tracking and management for RASC
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ExperimentTracker:
    """Track experiments with metrics, configs, and model checkpoints"""
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments/runs",
        config: Optional[Dict] = None
    ):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments
            config: Experiment configuration
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.experiment_name
        
        # Create experiment directory structure
        self.dirs = {
            'root': self.experiment_dir,
            'checkpoints': self.experiment_dir / 'checkpoints',
            'metrics': self.experiment_dir / 'metrics',
            'logs': self.experiment_dir / 'logs',
            'configs': self.experiment_dir / 'configs',
            'outputs': self.experiment_dir / 'outputs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking data
        self.metrics_history = {}
        self.config = config or {}
        self.metadata = {
            'name': experiment_name,
            'timestamp': timestamp,
            'start_time': datetime.now().isoformat()
        }
        
        # Save initial config
        if self.config:
            self.save_config(self.config)
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """
        Log a metric value
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
        """
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        metric_entry = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        if step is not None:
            metric_entry['step'] = step
        if epoch is not None:
            metric_entry['epoch'] = epoch
        
        self.metrics_history[name].append(metric_entry)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict] = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            epoch: Current epoch
            optimizer: Optimizer state
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {}
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.dirs['checkpoints'] / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.dirs['checkpoints'] / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)
        
        # Save latest
        latest_path = self.dirs['checkpoints'] / 'latest.pt'
        shutil.copy(checkpoint_path, latest_path)
    
    def save_config(self, config: Dict[str, Any], name: str = "config.json"):
        """Save configuration to file"""
        config_path = self.dirs['configs'] / name
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_metrics(self):
        """Save all metrics to file"""
        metrics_path = self.dirs['metrics'] / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def save_metadata(self):
        """Save experiment metadata"""
        self.metadata['end_time'] = datetime.now().isoformat()
        metadata_path = self.experiment_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def finalize(self):
        """Finalize experiment (save everything)"""
        self.save_metrics()
        self.save_metadata()
    
    def get_checkpoint_path(self, checkpoint_name: str = 'best_model.pt') -> Path:
        """Get path to a specific checkpoint"""
        return self.dirs['checkpoints'] / checkpoint_name
    
    def load_checkpoint(self, checkpoint_name: str = 'best_model.pt') -> Dict:
        """Load a checkpoint"""
        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path)


class MetricsTracker:
    """Simple metrics tracking without full experiment infrastructure"""
    
    def __init__(self):
        self.metrics = {}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], mode: str = 'train'):
        """Update metrics"""
        for name, value in metrics.items():
            full_name = f"{mode}_{name}"
            if full_name not in self.metrics:
                self.metrics[full_name] = []
            self.metrics[full_name].append(value)
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric"""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        return None
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average of a metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        values = self.metrics[name]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def is_best(self, name: str, value: float, mode: str = 'min') -> bool:
        """Check if this is the best value for a metric"""
        if name not in self.best_metrics:
            self.best_metrics[name] = value
            return True
        
        if mode == 'min':
            is_better = value < self.best_metrics[name]
        else:  # max
            is_better = value > self.best_metrics[name]
        
        if is_better:
            self.best_metrics[name] = value
        
        return is_better
