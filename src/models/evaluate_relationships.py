"""
Evaluate relationship prediction model on train/val/test splits
Outputs: Accuracy, F1-score, Confusion Matrix
Saves metrics to JSON files
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger
from models.relationship_models import create_model

logger = get_logger(__name__)

# Relationship types
RELATIONS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
]
rel2idx = {r: i for i, r in enumerate(RELATIONS)}
idx2rel = {i: r for r, i in rel2idx.items()}


class RelationshipDataset(Dataset):
    """Dataset for relationship prediction evaluation"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} relationship pairs from {data_path}")
    
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


def evaluate_relationship_model(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate model: Accuracy, F1, Confusion Matrix"""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating relationships"):
            s_cls, s_box, o_cls, o_box, y = [x.to(device) for x in batch]

            logits = model(s_cls, s_box, o_cls, o_box)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(len(RELATIONS))))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(RELATIONS))))

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": {idx2rel[i]: float(f1_per_class[i]) for i in range(len(RELATIONS))},
        "confusion_matrix": cm.tolist()
    }

    return metrics


def evaluate_split(model, device, config, data_path, batch_size, split_name, output_dir):
    """Evaluate a single split and save metrics to JSON"""
    dataset = RelationshipDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    metrics = evaluate_relationship_model(model, dataloader, device)

    output_file = output_dir / f"{split_name}_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {split_name} metrics to {output_file}")

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Relationship Prediction Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained relationship model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data folder containing train/val/test JSONs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs/eval_relationships", help="Where to save metrics")
    args = parser.parse_args()

    config = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using device: {device}")

    # Load model
    model_type = config.get("relationship.model_type", "neural_motifs")
    num_classes = config.get("relationship.num_classes", 150)
    num_relations = config.get("relationship.num_relations", 10)
    emb_dim = config.get("relationship.embedding_dim", 128)
    hidden_dim = config.get("relationship.architecture.hidden_dim", 512)
    dropout = config.get("relationship.architecture.dropout", 0.1)

    model = create_model(
        model_type,
        num_classes,
        num_relations,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate all splits
    splits = ["train", "val", "test"]
    for split in splits:
        split_file = Path(args.data_path) / f"{split}.json"
        if not split_file.exists():
            logger.warning(f"{split_file} does not exist, skipping")
            continue
        logger.info(f"Evaluating {split} split")
        metrics = evaluate_split(model, device, config, split_file, args.batch_size, split, output_dir)

        logger.info(f"{split.upper()} METRICS:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()



# """
# Evaluate relationship prediction model
# Outputs: Accuracy, F1-score, Confusion Matrix
# """

# import json
# from pathlib import Path
# from typing import Dict, List

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# from tqdm import tqdm

# import sys
# sys.path.append(str(Path(__file__).parent.parent))

# from utils.config import get_config
# from utils.logger import get_logger
# from models.relationship_models import create_model

# logger = get_logger(__name__)

# # Relationship types
# RELATIONS = [
#     "left of", "right of", "in front of", "behind",
#     "on top of", "under", "inside", "around", "over", "next to"
# ]
# rel2idx = {r: i for i, r in enumerate(RELATIONS)}
# idx2rel = {i: r for r, i in rel2idx.items()}


# class RelationshipDataset(Dataset):
#     """Dataset for relationship prediction evaluation"""
    
#     def __init__(self, data_path: str):
#         with open(data_path, 'r') as f:
#             self.data = json.load(f)
#         logger.info(f"Loaded {len(self.data)} relationship pairs")
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         d = self.data[idx]
#         return (
#             torch.tensor(d["subj_class"]),
#             torch.tensor(d["subj_bbox"], dtype=torch.float),
#             torch.tensor(d["obj_class"]),
#             torch.tensor(d["obj_bbox"], dtype=torch.float),
#             torch.tensor(rel2idx[d["predicate"]])
#         )


# def evaluate_relationship_model(
#     model,
#     dataloader: DataLoader,
#     device: torch.device
# ) -> Dict:
#     """Evaluate model: Accuracy, F1, Confusion Matrix"""

#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating relationships"):
#             s_cls, s_box, o_cls, o_box, y = [x.to(device) for x in batch]

#             logits = model(s_cls, s_box, o_cls, o_box)
#             preds = logits.argmax(dim=1)

#             all_preds.extend(preds.cpu().tolist())
#             all_labels.extend(y.cpu().tolist())

#     # Metrics
#     acc = accuracy_score(all_labels, all_preds)
#     f1_macro = f1_score(all_labels, all_preds, average='macro')
#     f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(len(RELATIONS))))
#     cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(RELATIONS))))

#     metrics = {
#         "accuracy": acc,
#         "f1_macro": f1_macro,
#         "f1_per_class": {idx2rel[i]: float(f1_per_class[i]) for i in range(len(RELATIONS))},
#         "confusion_matrix": cm.tolist()
#     }

#     return metrics


# def main():
#     import argparse

#     parser = argparse.ArgumentParser(description="Evaluate Relationship Prediction Model")
#     parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
#     parser.add_argument("--model-path", type=str, required=True, help="Path to trained relationship model")
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--data-path", type=str, default=None, help="JSON file with relationship pairs for evaluation")
#     args = parser.parse_args()

#     config = get_config(args.config)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # Load model
#     model_type = config.get("relationship.model_type", "neural_motifs")
#     num_classes = config.get("relationship.num_classes", 150)
#     num_relations = config.get("relationship.num_relations", 10)
#     emb_dim = config.get("relationship.embedding_dim", 128)
#     hidden_dim = config.get("relationship.architecture.hidden_dim", 512)
#     dropout = config.get("relationship.architecture.dropout", 0.1)

#     model = create_model(
#         model_type,
#         num_classes,
#         num_relations,
#         emb_dim=emb_dim,
#         hidden_dim=hidden_dim,
#         dropout=dropout
#     )
#     model.load_state_dict(torch.load(args.model_path, map_location=device))
#     model.to(device)

#     # Dataset
#     if args.data_path is None:
#         data_path = config.get("paths.relationship_pairs")
#     else:
#         data_path = args.data_path

#     dataset = RelationshipDataset(data_path)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

#     # Evaluate
#     metrics = evaluate_relationship_model(model, dataloader, device)

#     logger.info("Evaluation Complete!")
#     logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
#     logger.info(f"F1-Macro: {metrics['f1_macro']:.4f}")
#     logger.info("F1 per class:")
#     for rel, f1 in metrics['f1_per_class'].items():
#         logger.info(f"{rel}: {f1:.4f}")

#     logger.info("Confusion Matrix:")
#     for row in metrics["confusion_matrix"]:
#         logger.info(row)


# if __name__ == "__main__":
#     main()
