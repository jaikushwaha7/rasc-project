"""
Train relationship prediction model
Supports both MLP and Neural Motifs architectures
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger
from utils.experiment import ExperimentTracker, MetricsTracker
from models.relationship_models import create_model


logger = get_logger(__name__)


# Relationship types
RELATIONS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
]

rel2idx = {r: i for i, r in enumerate(RELATIONS)}


class RelationshipDataset(Dataset):
    """Dataset for relationship prediction"""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset
        
        Args:
            data_path: Path to relationship pairs JSON file
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} relationship pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.tensor(d["subj_class"]),
            torch.tensor(d["subj_bbox"], dtype=torch.float),
            torch.tensor(d["obj_class"]),
            torch.tensor(d["obj_bbox"], dtype=torch.float),
            torch.tensor(rel2idx[d["predicate"]])
        )


class RelationshipTrainer:
    """Trainer for relationship prediction models"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for this experiment
        """
        self.config = get_config(config_path)
        
        # Load configuration
        self.model_type = self.config.get('relationship.model_type', 'neural_motifs')
        self.num_classes = self.config.get('relationship.num_classes', 150)
        self.num_relations = self.config.get('relationship.num_relations', 10)
        
        # Training parameters
        self.epochs = self.config.get('relationship.training.epochs', 15)
        self.batch_size = self.config.get('relationship.training.batch_size', 64)
        self.learning_rate = self.config.get('relationship.training.learning_rate', 0.001)
        self.weight_decay = self.config.get('relationship.training.weight_decay', 0.0001)
        
        # Model parameters
        self.emb_dim = self.config.get('relationship.embedding_dim', 128)
        self.hidden_dim = self.config.get('relationship.architecture.hidden_dim', 512)
        self.dropout = self.config.get('relationship.architecture.dropout', 0.1)
        
        # Paths
        self.data_path = self.config.get('paths.relationship_pairs')
        self.output_dir = Path(self.config.get('paths.models')) / 'relationship_predictor'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        if experiment_name is None:
            experiment_name = f"relationship_{self.model_type}"
        
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            base_dir=self.config.get('paths.experiments'),
            config=self._get_training_config()
        )
        
        self.metrics_tracker = MetricsTracker()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized relationship trainer: {experiment_name}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Device: {self.device}")
    
    def _get_training_config(self) -> Dict:
        """Get training configuration for logging"""
        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "num_relations": self.num_relations,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "embedding_dim": self.emb_dim,
            "hidden_dim": self.hidden_dim
        }
    
    def create_data_loaders(self):
        """Create train and validation data loaders"""
        logger.info("Creating data loaders...")
        
        dataset = RelationshipDataset(self.data_path)
        
        # Split into train and validation
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self):
        """Train the relationship model"""
        logger.info("="*60)
        logger.info(f"Starting relationship model training")
        logger.info("="*60)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        
        # Create model
        logger.info(f"Creating {self.model_type} model...")
        model = create_model(
            self.model_type,
            self.num_classes,
            self.num_relations,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        model = model.to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Train
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion, epoch)
            
            # Log metrics
            epoch_metrics = {
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy']
            }
            
            # self.tracker.log_epoch(epoch, epoch_metrics)
            self.tracker.log_metrics(epoch_metrics, epoch=epoch)            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                self._save_model(model, epoch, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_model(model, epoch)
        
        # Finalize
        self.tracker.finalize()
        
        logger.info("="*60)
        logger.info("Training completed successfully")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("="*60)
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict:
        """Train for one epoch"""
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            s_cls, s_box, o_cls, o_box, y = [x.to(self.device) for x in batch]
            
            # Forward pass
            logits = model(s_cls, s_box, o_cls, o_box)
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        logger.info(
            f"Epoch {epoch} [Train] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Dict:
        """Validate the model"""
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
            
            for batch in pbar:
                s_cls, s_box, o_cls, o_box, y = [x.to(self.device) for x in batch]
                
                logits = model(s_cls, s_box, o_cls, o_box)
                loss = criterion(logits, y)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        logger.info(
            f"Epoch {epoch} [Val] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _save_model(self, model: nn.Module, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f'{self.model_type}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / f'{self.model_type}_best.pt'
            torch.save(model.state_dict(), best_path)
            logger.info(f"Saved best model to {best_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train relationship prediction model"
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
    
    args = parser.parse_args()
    
    trainer = RelationshipTrainer(
        config_path=args.config,
        experiment_name=args.experiment_name
    )
    trainer.train()


if __name__ == "__main__":
    main()
