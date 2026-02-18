"""
Train YOLOv8 object detection model on Visual Genome
Includes experiment tracking, logging, and model checkpointing
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger
from utils.experiment import ExperimentTracker


logger = get_logger(__name__)


class YOLOTrainer:
    """YOLO model trainer with experiment tracking"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize YOLO trainer
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for this experiment
        """
        self.config = get_config(config_path)
        
        # Load training configuration
        self.model_name = self.config.get('detection.model_name', 'yolov8n')
        self.data_yaml = "configs/yolo.yaml"
        
        # Training parameters
        self.epochs = self.config.get('detection.training.epochs', 10)
        self.batch_size = self.config.get('detection.training.batch_size', 16)
        self.img_size = self.config.get('detection.training.image_size', [512, 640])
        self.workers = self.config.get('detection.training.workers', 4)
        self.device = self.config.get('detection.training.device', 'cpu')
        
        # Optimizer parameters
        self.lr0 = self.config.get('detection.optimizer.lr0', 0.01)
        
        # Experiment tracking
        if experiment_name is None:
            experiment_name = f"yolo_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            base_dir=self.config.get('paths.experiments'),
            config=self._get_training_config()
        )
        
        logger.info(f"Initialized YOLO trainer: {experiment_name}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Epochs: {self.epochs}, Batch size: {self.batch_size}")
        logger.info(f"Image size: {self.img_size}")
        logger.info(f"Device: {self.device}")
    
    def _get_training_config(self) -> Dict:
        """Get training configuration for logging"""
        return {
            "model": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.img_size,
            "optimizer": "AdamW",
            "learning_rate": self.lr0,
            "dataset": "VG-5K",
            "device": self.device
        }
    
    def train(self):
        """Train YOLO model"""
        logger.info("="*60)
        logger.info(f"Starting YOLO training: {self.experiment_name}")
        logger.info("="*60)
        
        # Initialize model
        logger.info(f"Loading pretrained {self.model_name} model...")
        model = YOLO(f"{self.model_name}.pt")
        
        # Train
        logger.info("Starting training...")
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=tuple(self.img_size) if isinstance(self.img_size, list) else self.img_size,
            batch=self.batch_size,
            project=str(self.tracker.dirs['root'].parent),
            name=self.experiment_name,
            pretrained=True,
            lr0=self.lr0,
            workers=self.workers,
            device=self.device,
            exist_ok=True,
            verbose=True,
            plots=True,
            save=True,
            save_period=5  # Save checkpoint every 5 epochs
        )
        
        # Extract and log metrics
        logger.info("Training completed. Extracting metrics...")
        metrics = self._extract_metrics(results)
        self._log_metrics(metrics)
        
        # Save final results
        self._save_results(results, metrics)
        
        logger.info("="*60)
        logger.info("YOLO training completed successfully")
        logger.info(f"Results saved to: {self.tracker.dirs['root']}")
        logger.info("="*60)
        
        return results
    
    def _extract_metrics(self, results) -> Dict:
        """Extract metrics from training results"""
        metrics = {}
        
        try:
            # Box metrics
            # if hasattr(results, 'metrics') and hasattr(results.metrics, 'box'):
            #     metrics['mAP50'] = float(results.metrics.box.map50)
            #     metrics['mAP50_95'] = float(results.metrics.box.map)
                
            #     # Additional metrics if available
            #     if hasattr(results.metrics.box, 'mp'):
            #         metrics['precision'] = float(results.metrics.box.mp)
            #     if hasattr(results.metrics.box, 'mr'):
            #         metrics['recall'] = float(results.metrics.box.mr)
            # FIX: Access .box directly from results
            if hasattr(results, 'box'):
                metrics['mAP50'] = float(results.box.map50)
                metrics['mAP50_95'] = float(results.box.map)
                
                if hasattr(results.box, 'mp'):
                    metrics['precision'] = float(results.box.mp)
                if hasattr(results.box, 'mr'):
                    metrics['recall'] = float(results.box.mr)



            logger.info("Extracted metrics:")
            for name, value in metrics.items():
                logger.info(f"  {name}: {value:.4f}")
        
        except Exception as e:
            logger.warning(f"Error extracting metrics: {e}")
        
        return metrics
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to experiment tracker"""
        for name, value in metrics.items():
            self.tracker.log_metric(name, value)
    
    def _save_results(self, results, metrics: Dict):
        """Save training results and metrics"""
        # Save metrics JSON
        metrics_file = self.tracker.dirs['metrics'] / 'final_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Finalize experiment
        self.tracker.finalize()
    
    def evaluate(self, weights_path: Optional[str] = None):
        """
        Evaluate trained model on test set
        
        Args:
            weights_path: Path to model weights (uses best if None)
        """
        if weights_path is None:
            # Find best weights from YOLO training
            weights_path = self.tracker.dirs['root'] / 'weights' / 'best.pt'
        
        logger.info(f"Evaluating model: {weights_path}")
        
        model = YOLO(str(weights_path))
        results = model.val(
            data=self.data_yaml,
            split='test'
        )
        
        logger.info("Evaluation completed")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 object detection model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for this experiment'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights for evaluation'
    )
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(
        config_path=args.config,
        experiment_name=args.experiment_name
    )
    
    if args.eval_only:
        trainer.evaluate(weights_path=args.weights)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
