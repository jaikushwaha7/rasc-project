"""
End-to-end inference for RASC
UNIVERSAL VERSION: Works with both MLP and Neural Motifs models
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import inspect

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import T5Tokenizer, T5ForConditionalGeneration

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger
from models.relationship_models import create_model


logger = get_logger(__name__)

RELATIONS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "under", "inside", "around", "over", "next to"
]


class RASCInference:
    """End-to-end inference pipeline for RASC - Universal version"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        yolo_weights: Optional[str] = None,
        relationship_weights: Optional[str] = None,
        caption_model: Optional[str] = None
    ):
        """
        Initialize inference pipeline
        
        Args:
            config_path: Path to configuration file
            yolo_weights: Path to YOLO weights
            relationship_weights: Path to relationship model weights
            caption_model: Path to caption model directory
        """
        self.config = get_config(config_path)
        
        # Load models
        logger.info("Loading models...")
        self._load_models(yolo_weights, relationship_weights, caption_model)
        
        # Configuration
        self.max_objects = self.config.get('detection.inference.max_detections', 15)
        self.conf_threshold = self.config.get('detection.inference.conf_threshold', 0.25)
        self.img_size = self.config.get('detection.inference.image_size', (512, 640))
        
        logger.info("Inference pipeline ready")
    
    def _load_models(
        self,
        yolo_weights: Optional[str],
        relationship_weights: Optional[str],
        caption_model: Optional[str]
    ):
        """Load all required models"""
        
        # YOLO detector
        if yolo_weights is None:
            yolo_weights = "yolov8n.pt"
        
        logger.info(f"Loading YOLO from {yolo_weights}")
        self.yolo = YOLO(yolo_weights)
        
        # Relationship model
        if relationship_weights is None:
            relationship_weights = Path(self.config.get('paths.models')) / \
                'relationship_predictor' / 'neural_motifs_best.pt'
        
        logger.info(f"Loading relationship model from {relationship_weights}")
        
        model_type = self.config.get('relationship.model_type', 'neural_motifs')
        num_classes = self.config.get('relationship.num_classes', 150)
        num_relations = self.config.get('relationship.num_relations', 10)
        
        self.rel_model = create_model(
            model_type,
            num_classes,
            num_relations,
            emb_dim=self.config.get('relationship.embedding_dim', 128)
        )
        
        # Load weights with proper error handling
        if Path(relationship_weights).exists():
            self.rel_model.load_state_dict(
                torch.load(relationship_weights, map_location='cpu'),
                strict=False  # Allow partial loading
            )
            logger.info("Loaded relationship model weights")
        else:
            logger.warning(f"Relationship weights not found at {relationship_weights}")
        
        self.rel_model.eval()
        
        # Detect model type by checking forward signature
        self._detect_model_type()
        
        # Caption generator
        if caption_model is None:
            caption_model = Path(self.config.get('paths.models')) / \
                'caption_generator' / 't5_scene'
        
        logger.info(f"Loading caption model from {caption_model}")
        
        if Path(caption_model).exists():
            self.tokenizer = T5Tokenizer.from_pretrained(str(caption_model))
            self.caption_model = T5ForConditionalGeneration.from_pretrained(str(caption_model))
        else:
            logger.warning(f"Caption model not found, using pretrained T5")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.caption_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        self.caption_model.eval()
    
    def _detect_model_type(self):
        """Detect whether the loaded model is MLP or Neural Motifs"""
        forward_sig = inspect.signature(self.rel_model.forward)
        params = list(forward_sig.parameters.keys())
        
        # MLP has 4 params: (s_cls, s_box, o_cls, o_box)
        # Neural Motifs has 3 params: (obj_classes, obj_boxes, pair_idx)
        
        if len(params) == 4:
            self.model_type = 'mlp'
            logger.info("Detected MLP model (4 parameters)")
        elif len(params) == 3:
            self.model_type = 'neural_motifs'
            logger.info("Detected Neural Motifs model (3 parameters)")
        else:
            # Fallback to config
            self.model_type = self.config.get('relationship.model_type', 'neural_motifs')
            logger.warning(f"Could not auto-detect model type, using config: {self.model_type}")
    
    def detect_objects(self, image_path: str) -> List[Tuple[int, torch.Tensor]]:
        """
        Detect objects in image
        
        Args:
            image_path: Path to image
            
        Returns:
            List of (class_id, bbox) tuples where bbox is [x, y, w, h] normalized
        """
        logger.info(f"Detecting objects in {image_path}")
        
        start_time = time.time()
        results = self.yolo(
            image_path,
            imgsz=self.img_size,
            conf=self.conf_threshold
        )[0]
        
        detection_time = time.time() - start_time
        
        objects = []
        for cls, box, conf in zip(results.boxes.cls, results.boxes.xywh, results.boxes.conf):
            if len(objects) >= self.max_objects:
                break
            
            # Normalize bbox coordinates
            normalized_box = box / max(self.img_size)
            objects.append((int(cls), normalized_box))
        
        logger.info(
            f"Detected {len(objects)} objects in {detection_time*1000:.1f}ms"
        )
        
        return objects
    
    def predict_relationships(
        self,
        objects: List[Tuple[int, torch.Tensor]]
    ) -> List[str]:
        """
        Predict relationships between objects
        UNIVERSAL: Works with both MLP and Neural Motifs
        
        Args:
            objects: List of (class_id, bbox) tuples
            
        Returns:
            List of relationship triplets as strings
        """
        if len(objects) < 2:
            logger.warning("Not enough objects for relationship prediction")
            return []
        
        logger.info(f"Predicting relationships for {len(objects)} objects using {self.model_type} model")
        
        # Generate all pairs
        pairs = []
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i != j:
                    pairs.append([i, j])
        
        if len(pairs) == 0:
            return []
        
        relationships = []
        
        if self.model_type == 'mlp':
            # MLP model: predict each pair individually
            relationships = self._predict_mlp(objects, pairs)
        else:
            # Neural Motifs: batch prediction
            relationships = self._predict_neural_motifs(objects, pairs)
        
        # Limit to avoid too long sequences
        relationships = relationships[:40]
        
        logger.info(f"Predicted {len(relationships)} relationships")
        
        return relationships
    
    def _predict_mlp(
        self,
        objects: List[Tuple[int, torch.Tensor]],
        pairs: List[List[int]]
    ) -> List[str]:
        """Predict using MLP model (per-pair prediction)"""
        
        relationships = []
        
        with torch.no_grad():
            for i, j in pairs:
                s_cls, s_box = objects[i]
                o_cls, o_box = objects[j]
                
                # Ensure boxes are the right shape [1, 4]
                if s_box.dim() == 1:
                    s_box = s_box.unsqueeze(0)
                if o_box.dim() == 1:
                    o_box = o_box.unsqueeze(0)
                
                # Convert to tensors
                s_cls_tensor = torch.tensor([s_cls], dtype=torch.long)
                o_cls_tensor = torch.tensor([o_cls], dtype=torch.long)
                
                try:
                    # MLP forward: (s_cls, s_box, o_cls, o_box)
                    logits = self.rel_model(s_cls_tensor, s_box, o_cls_tensor, o_box)
                    pred = logits.argmax().item()
                    
                    rel_text = f"obj{i} {RELATIONS[pred]} obj{j}"
                    relationships.append(rel_text)
                    
                except Exception as e:
                    logger.error(f"Error predicting relationship for pair ({i}, {j}): {e}")
                    continue
        
        return relationships
    
    def _predict_neural_motifs(
        self,
        objects: List[Tuple[int, torch.Tensor]],
        pairs: List[List[int]]
    ) -> List[str]:
        """Predict using Neural Motifs model (batch prediction)"""
        
        # Prepare inputs
        obj_classes = torch.tensor([o[0] for o in objects], dtype=torch.long)
        
        # Stack bboxes properly - ensure all have same shape
        obj_boxes_list = []
        for o in objects:
            bbox = o[1]
            # Ensure bbox is 1D tensor with 4 elements
            if bbox.dim() == 0:
                bbox = bbox.unsqueeze(0)
            if bbox.dim() > 1:
                bbox = bbox.squeeze()
            
            # Ensure exactly 4 elements
            if bbox.shape[0] != 4:
                logger.warning(f"Unexpected bbox shape: {bbox.shape}, fixing to 4")
                if bbox.shape[0] < 4:
                    # Pad with zeros
                    bbox = torch.cat([bbox, torch.zeros(4 - bbox.shape[0])])
                else:
                    # Truncate
                    bbox = bbox[:4]
            
            obj_boxes_list.append(bbox)
        
        # Stack into [N, 4] tensor
        obj_boxes = torch.stack(obj_boxes_list)
        
        # Verify shapes
        assert obj_classes.shape[0] == obj_boxes.shape[0], \
            f"Mismatch: classes={obj_classes.shape[0]}, boxes={obj_boxes.shape[0]}"
        assert obj_boxes.shape[1] == 4, \
            f"Boxes should have 4 coords, got {obj_boxes.shape[1]}"
        
        pair_idx = torch.tensor(pairs, dtype=torch.long)
        
        # Predict
        relationships = []
        
        with torch.no_grad():
            try:
                # Neural Motifs forward: (obj_classes, obj_boxes, pair_idx)
                logits = self.rel_model(obj_classes, obj_boxes, pair_idx)
                preds = logits.argmax(dim=1)
                
                # Format as triplets
                for (i, j), p in zip(pairs, preds):
                    rel_text = f"obj{i} {RELATIONS[p]} obj{j}"
                    relationships.append(rel_text)
                    
            except Exception as e:
                logger.error(f"Error in Neural Motifs prediction: {e}")
                logger.error(f"obj_classes shape: {obj_classes.shape}")
                logger.error(f"obj_boxes shape: {obj_boxes.shape}")
                logger.error(f"pair_idx shape: {pair_idx.shape}")
                raise
        
        return relationships
    
    def generate_caption(self, relationships: List[str]) -> str:
        """
        Generate caption from scene graph
        
        Args:
            relationships: List of relationship triplets
            
        Returns:
            Generated caption
        """
        if not relationships:
            logger.warning("No relationships provided for caption generation")
            return "No objects or relationships detected in the image."
        
        logger.info("Generating caption from scene graph")
        
        # Create scene graph text
        graph_text = "; ".join(relationships)
        
        # Tokenize
        inputs = self.tokenizer(
            graph_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.caption_model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated caption: {caption}")
        
        return caption
    
    def run(self, image_path: str, verbose: bool = True) -> Dict:
        """
        Run full inference pipeline
        
        Args:
            image_path: Path to input image
            verbose: Print detailed output
            
        Returns:
            Dictionary with results
        """
        logger.info("="*60)
        logger.info(f"Running inference on {image_path}")
        logger.info("="*60)
        
        total_start = time.time()
        
        # 1. Object detection
        objects = self.detect_objects(image_path)
        
        # 2. Relationship prediction
        relationships = self.predict_relationships(objects)
        
        # 3. Caption generation
        caption = self.generate_caption(relationships)
        
        total_time = time.time() - total_start
        
        results = {
            'image_path': image_path,
            'num_objects': len(objects),
            'num_relationships': len(relationships),
            'relationships': relationships,
            'caption': caption,
            'inference_time': total_time,
            'model_type': self.model_type
        }
        
        if verbose:
            logger.info("="*60)
            logger.info("RESULTS")
            logger.info("="*60)
            logger.info(f"Model type: {results['model_type']}")
            logger.info(f"Objects detected: {results['num_objects']}")
            logger.info(f"Relationships: {results['num_relationships']}")
            logger.info(f"Caption: {results['caption']}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info("="*60)
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RASC Inference - Generate relationship-aware captions"
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--yolo-weights',
        type=str,
        default=None,
        help='Path to YOLO weights'
    )
    
    parser.add_argument(
        '--relationship-weights',
        type=str,
        default=None,
        help='Path to relationship model weights'
    )
    
    parser.add_argument(
        '--caption-model',
        type=str,
        default=None,
        help='Path to caption model directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RASCInference(
        config_path=args.config,
        yolo_weights=args.yolo_weights,
        relationship_weights=args.relationship_weights,
        caption_model=args.caption_model
    )
    
    # Run inference
    results = pipeline.run(args.image)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
# """
# End-to-end inference for RASC
# Performs object detection → relationship prediction → caption generation
# """

# import time
# import json
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional

# import torch
# from ultralytics import YOLO
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# import sys
# sys.path.append(str(Path(__file__).parent.parent))

# from utils.config import get_config
# from utils.logger import get_logger
# from models.relationship_models import create_model


# logger = get_logger(__name__)

# RELATIONS = [
#     "left of", "right of", "in front of", "behind",
#     "on top of", "under", "inside", "around", "over", "next to"
# ]


# class RASCInference:
#     """End-to-end inference pipeline for RASC"""

#     def __init__(
#         self,
#         config_path: Optional[str] = None,
#         yolo_weights: Optional[str] = None,
#         relationship_weights: Optional[str] = None,
#         caption_model: Optional[str] = None
#     ):
#         self.config = get_config(config_path)

#         # Device setup
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         logger.info(f"Using device: {self.device}")

#         logger.info("Loading models...")
#         self._load_models(yolo_weights, relationship_weights, caption_model)

#         self.max_objects = self.config.get('detection.inference.max_detections', 15)
#         self.conf_threshold = self.config.get('detection.inference.conf_threshold', 0.25)
#         self.img_size = self.config.get('detection.inference.image_size', 640)

#         logger.info("Inference pipeline ready")

#     # ---------------------------------------------------
#     # Model Loading
#     # ---------------------------------------------------

#     def _load_models(self, yolo_weights, relationship_weights, caption_model):

#         # YOLO
#         if yolo_weights is None:
#             yolo_weights = "yolov8n.pt"

#         logger.info(f"Loading YOLO from {yolo_weights}")
#         self.yolo = YOLO(yolo_weights)
#         self.yolo.to(self.device)

#         # Relationship Model
#         if relationship_weights is None:
#             relationship_weights = Path(self.config.get('paths.models')) / \
#                 'relationship_predictor' / 'neural_motifs.pt'

#         logger.info(f"Loading relationship model from {relationship_weights}")

#         model_type = self.config.get('relationship.model_type', 'neural_motifs')
#         num_classes = self.config.get('relationship.num_classes', 150)
#         num_relations = self.config.get('relationship.num_relations', 10)

#         self.rel_model = create_model(
#             model_type,
#             num_classes,
#             num_relations,
#             emb_dim=self.config.get('relationship.embedding_dim', 128)
#         )

#         self.rel_model.load_state_dict(
#             torch.load(relationship_weights, map_location=self.device)
#         )

#         self.rel_model.to(self.device)
#         self.rel_model.eval()

#         # Caption Model
#         if caption_model is None:
#             caption_model = Path(self.config.get('paths.models')) / \
#                 'caption_generator' / 't5_scene'

#         logger.info(f"Loading caption model from {caption_model}")

#         self.tokenizer = T5Tokenizer.from_pretrained(str(caption_model))
#         self.caption_model = T5ForConditionalGeneration.from_pretrained(str(caption_model))
#         self.caption_model.to(self.device)
#         self.caption_model.eval()

#     # ---------------------------------------------------
#     # Object Detection
#     # ---------------------------------------------------

#     def detect_objects(self, image_path: str):

#         logger.info(f"Detecting objects in {image_path}")

#         start_time = time.time()

#         results = self.yolo(
#             image_path,
#             imgsz=self.img_size,
#             conf=self.conf_threshold,
#             device=self.device
#         )[0]

#         detection_time = time.time() - start_time

#         objects = []
#         for cls, box in zip(results.boxes.cls, results.boxes.xywh):

#             if len(objects) >= self.max_objects:
#                 break

#             normalized_box = box.to(self.device) / self.img_size
#             objects.append((int(cls.item()), normalized_box))

#         logger.info(
#             f"Detected {len(objects)} objects in {detection_time*1000:.1f}ms"
#         )

#         return objects

#     # ---------------------------------------------------
#     # Relationship Prediction
#     # ---------------------------------------------------

#     def predict_relationships(self, objects):

#         if len(objects) < 2:
#             logger.warning("Not enough objects for relationship prediction")
#             return []

#         logger.info(f"Predicting relationships for {len(objects)} objects")

#         obj_classes = torch.tensor(
#             [o[0] for o in objects],
#             dtype=torch.long,
#             device=self.device
#         ).view(-1)

#         obj_boxes = torch.stack(
#             [o[1] for o in objects]
#         ).to(self.device)

#         obj_boxes = obj_boxes.view(-1, 4)

#         # Generate object pairs
#         pairs = []
#         for i in range(len(objects)):
#             for j in range(len(objects)):
#                 if i != j:
#                     pairs.append((i, j))

#         if len(pairs) == 0:
#             return []

#         pair_idx = torch.tensor(pairs, device=self.device)

#         # Build object boxes per pair
#         s_boxes = obj_boxes[pair_idx[:, 0]]  # subject boxes
#         o_boxes = obj_boxes[pair_idx[:, 1]]  # object boxes

#         with torch.no_grad():
#             logits = self.rel_model(
#                 obj_classes,
#                 obj_boxes,
#                 pair_idx,
#                 o_boxes   # correct shape (num_pairs, 4)
#             )
#             preds = logits.argmax(dim=1)

#         relationships = []

#         for (i, j), p in zip(pairs, preds):
#             rel_text = f"obj{i} {RELATIONS[p.item()]} obj{j}"
#             relationships.append(rel_text)

#         relationships = relationships[:40]

#         logger.info(f"Predicted {len(relationships)} relationships")

#         return relationships

#     # ---------------------------------------------------
#     # Caption Generation
#     # ---------------------------------------------------

#     def generate_caption(self, relationships):

#         if not relationships:
#             return "No objects or relationships detected in the image."

#         graph_text = "; ".join(relationships)

#         inputs = self.tokenizer(
#             graph_text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512
#         ).to(self.device)

#         with torch.no_grad():
#             outputs = self.caption_model.generate(
#                 **inputs,
#                 max_length=64,
#                 num_beams=4,
#                 early_stopping=True
#             )

#         caption = self.tokenizer.decode(
#             outputs[0],
#             skip_special_tokens=True
#         )

#         return caption

#     # ---------------------------------------------------
#     # Full Pipeline
#     # ---------------------------------------------------

#     def run(self, image_path: str, verbose: bool = True):

#         logger.info("=" * 60)
#         logger.info(f"Running inference on {image_path}")
#         logger.info("=" * 60)

#         total_start = time.time()

#         objects = self.detect_objects(image_path)
#         relationships = self.predict_relationships(objects)
#         caption = self.generate_caption(relationships)

#         total_time = time.time() - total_start

#         results = {
#             'image_path': image_path,
#             'num_objects': len(objects),
#             'num_relationships': len(relationships),
#             'relationships': relationships,
#             'caption': caption,
#             'inference_time': total_time
#         }

#         if verbose:
#             logger.info("=" * 60)
#             logger.info("RESULTS")
#             logger.info("=" * 60)
#             logger.info(f"Objects detected: {results['num_objects']}")
#             logger.info(f"Relationships: {results['num_relationships']}")
#             logger.info(f"Caption: {results['caption']}")
#             logger.info(f"Total time: {total_time:.2f}s")
#             logger.info("=" * 60)

#         return results


# # ---------------------------------------------------
# # CLI
# # ---------------------------------------------------

# def main():

#     import argparse

#     parser = argparse.ArgumentParser(
#         description="RASC Inference - Generate relationship-aware captions"
#     )

#     parser.add_argument('--image', type=str, required=True)
#     parser.add_argument('--config', type=str, default=None)
#     parser.add_argument('--yolo-weights', type=str, default=None)
#     parser.add_argument('--relationship-weights', type=str, default=None)
#     parser.add_argument('--caption-model', type=str, default=None)
#     parser.add_argument('--output', type=str, default=None)

#     args = parser.parse_args()

#     pipeline = RASCInference(
#         config_path=args.config,
#         yolo_weights=args.yolo_weights,
#         relationship_weights=args.relationship_weights,
#         caption_model=args.caption_model
#     )

#     results = pipeline.run(args.image)

#     if args.output:
#         with open(args.output, 'w') as f:
#             json.dump(results, f, indent=2)
#         logger.info(f"Results saved to {args.output}")


# if __name__ == "__main__":
#     main()

# # """
# # End-to-end inference for RASC
# # Performs object detection → relationship prediction → caption generation
# # """

# # import time
# # from pathlib import Path
# # from typing import Dict, List, Tuple, Optional

# # import torch
# # from PIL import Image
# # from ultralytics import YOLO
# # from transformers import T5Tokenizer, T5ForConditionalGeneration

# # import sys
# # sys.path.append(str(Path(__file__).parent.parent))

# # from utils.config import get_config
# # from utils.logger import get_logger
# # from models.relationship_models import create_model


# # logger = get_logger(__name__)

# # RELATIONS = [
# #     "left of", "right of", "in front of", "behind",
# #     "on top of", "under", "inside", "around", "over", "next to"
# # ]


# # class RASCInference:
# #     """End-to-end inference pipeline for RASC"""
    
# #     def __init__(
# #         self,
# #         config_path: Optional[str] = None,
# #         yolo_weights: Optional[str] = None,
# #         relationship_weights: Optional[str] = None,
# #         caption_model: Optional[str] = None
# #     ):
# #         """
# #         Initialize inference pipeline
        
# #         Args:
# #             config_path: Path to configuration file
# #             yolo_weights: Path to YOLO weights
# #             relationship_weights: Path to relationship model weights
# #             caption_model: Path to caption model directory
# #         """
# #         self.config = get_config(config_path)
        
# #         # Load models
# #         logger.info("Loading models...")
# #         self._load_models(yolo_weights, relationship_weights, caption_model)
        
# #         # Configuration
# #         self.max_objects = self.config.get('detection.inference.max_detections', 15)
# #         self.conf_threshold = self.config.get('detection.inference.conf_threshold', 0.25)
# #         self.img_size = self.config.get('detection.inference.image_size', (512, 640))
        
# #         logger.info("Inference pipeline ready")
    
# #     def _load_models(
# #         self,
# #         yolo_weights: Optional[str],
# #         relationship_weights: Optional[str],
# #         caption_model: Optional[str]
# #     ):
# #         """Load all required models"""
        
# #         # YOLO detector
# #         if yolo_weights is None:
# #             yolo_weights = "yolov8n.pt"  # Default pretrained
        
# #         logger.info(f"Loading YOLO from {yolo_weights}")
# #         self.yolo = YOLO(yolo_weights)
        
# #         # Relationship model
# #         if relationship_weights is None:
# #             relationship_weights = Path(self.config.get('paths.models')) / \
# #                 'relationship_predictor' / 'neural_motifs.pt'
        
# #         logger.info(f"Loading relationship model from {relationship_weights}")
        
# #         model_type = self.config.get('relationship.model_type', 'neural_motifs')
# #         num_classes = self.config.get('relationship.num_classes', 150)
# #         num_relations = self.config.get('relationship.num_relations', 10)
        
# #         self.rel_model = create_model(
# #             model_type,
# #             num_classes,
# #             num_relations,
# #             emb_dim=self.config.get('relationship.embedding_dim', 128)
# #         )
        
# #         self.rel_model.load_state_dict(torch.load(relationship_weights, map_location='cpu'))
# #         self.rel_model.eval()
        
# #         # Caption generator
# #         if caption_model is None:
# #             caption_model = Path(self.config.get('paths.models')) / \
# #                 'caption_generator' / 't5_scene'
        
# #         logger.info(f"Loading caption model from {caption_model}")
# #         self.tokenizer = T5Tokenizer.from_pretrained(str(caption_model))
# #         self.caption_model = T5ForConditionalGeneration.from_pretrained(str(caption_model))
# #         self.caption_model.eval()
    
# #     def detect_objects(self, image_path: str) -> List[Tuple[int, torch.Tensor]]:
# #         """
# #         Detect objects in image
        
# #         Args:
# #             image_path: Path to image
            
# #         Returns:
# #             List of (class_id, bbox) tuples
# #         """
# #         logger.info(f"Detecting objects in {image_path}")
        
# #         start_time = time.time()
# #         results = self.yolo(
# #             image_path,
# #             imgsz=self.img_size,
# #             conf=self.conf_threshold
# #         )[0]
        
# #         detection_time = time.time() - start_time
        
# #         objects = []
# #         for cls, box, conf in zip(results.boxes.cls, results.boxes.xywh, results.boxes.conf):
# #             if len(objects) >= self.max_objects:
# #                 break
            
# #             # Normalize bbox coordinates
# #             normalized_box = box / max(self.img_size)
# #             objects.append((int(cls), normalized_box))
        
# #         logger.info(
# #             f"Detected {len(objects)} objects in {detection_time*1000:.1f}ms"
# #         )
        
# #         return objects
    
# #     def predict_relationships(
# #         self,
# #         objects: List[Tuple[int, torch.Tensor]]
# #     ) -> List[str]:
# #         """
# #         Predict relationships between objects
        
# #         Args:
# #             objects: List of (class_id, bbox) tuples
            
# #         Returns:
# #             List of relationship triplets
# #         """
# #         if len(objects) < 2:
# #             logger.warning("Not enough objects for relationship prediction")
# #             return []
        
# #         logger.info(f"Predicting relationships for {len(objects)} objects")
        
# #         # Prepare inputs
# #         obj_classes = torch.tensor([o[0] for o in objects])
# #         obj_boxes = torch.stack([o[1] for o in objects])
        
# #         # Generate all pairs
# #         pairs = []
# #         for i in range(len(objects)):
# #             for j in range(len(objects)):
# #                 if i != j:
# #                     pairs.append([i, j])
        
# #         if len(pairs) == 0:
# #             return []
        
# #         pair_idx = torch.tensor(pairs)
        
# #         # Predict
# #         with torch.no_grad():
# #             logits = self.rel_model(obj_classes, obj_boxes, pair_idx, obj_boxes)
# #             # logits = self.rel_model(obj_classes, obj_boxes, pair_idx)
# #             preds = logits.argmax(dim=1)
        
# #         # Format as triplets
# #         relationships = []
# #         for (i, j), p in zip(pairs, preds):
# #             rel_text = f"obj{i} {RELATIONS[p]} obj{j}"
# #             relationships.append(rel_text)
        
# #         # Limit to avoid too long sequences
# #         relationships = relationships[:40]
        
# #         logger.info(f"Predicted {len(relationships)} relationships")
        
# #         return relationships
    
# #     def generate_caption(self, relationships: List[str]) -> str:
# #         """
# #         Generate caption from scene graph
        
# #         Args:
# #             relationships: List of relationship triplets
            
# #         Returns:
# #             Generated caption
# #         """
# #         if not relationships:
# #             logger.warning("No relationships provided for caption generation")
# #             return "No objects or relationships detected in the image."
        
# #         logger.info("Generating caption from scene graph")
        
# #         # Create scene graph text
# #         graph_text = "; ".join(relationships)
        
# #         # Tokenize
# #         inputs = self.tokenizer(
# #             graph_text,
# #             return_tensors="pt",
# #             truncation=True,
# #             max_length=512
# #         )
        
# #         # Generate
# #         with torch.no_grad():
# #             outputs = self.caption_model.generate(
# #                 **inputs,
# #                 max_length=64,
# #                 num_beams=4,
# #                 early_stopping=True
# #             )
        
# #         # Decode
# #         caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
# #         logger.info(f"Generated caption: {caption}")
        
# #         return caption
    
# #     def run(self, image_path: str, verbose: bool = True) -> Dict:
# #         """
# #         Run full inference pipeline
        
# #         Args:
# #             image_path: Path to input image
# #             verbose: Print detailed output
            
# #         Returns:
# #             Dictionary with results
# #         """
# #         logger.info("="*60)
# #         logger.info(f"Running inference on {image_path}")
# #         logger.info("="*60)
        
# #         total_start = time.time()
        
# #         # 1. Object detection
# #         objects = self.detect_objects(image_path)
        
# #         # 2. Relationship prediction
# #         relationships = self.predict_relationships(objects)
        
# #         # 3. Caption generation
# #         caption = self.generate_caption(relationships)
        
# #         total_time = time.time() - total_start
        
# #         results = {
# #             'image_path': image_path,
# #             'num_objects': len(objects),
# #             'num_relationships': len(relationships),
# #             'relationships': relationships,
# #             'caption': caption,
# #             'inference_time': total_time
# #         }
        
# #         if verbose:
# #             logger.info("="*60)
# #             logger.info("RESULTS")
# #             logger.info("="*60)
# #             logger.info(f"Objects detected: {results['num_objects']}")
# #             logger.info(f"Relationships: {results['num_relationships']}")
# #             logger.info(f"Caption: {results['caption']}")
# #             logger.info(f"Total time: {total_time:.2f}s")
# #             logger.info("="*60)
        
# #         return results


# # def main():
# #     """Main entry point"""
# #     import argparse
    
# #     parser = argparse.ArgumentParser(
# #         description="RASC Inference - Generate relationship-aware captions"
# #     )
    
# #     parser.add_argument(
# #         '--image',
# #         type=str,
# #         required=True,
# #         help='Path to input image'
# #     )
    
# #     parser.add_argument(
# #         '--config',
# #         type=str,
# #         default=None,
# #         help='Path to configuration file'
# #     )
    
# #     parser.add_argument(
# #         '--yolo-weights',
# #         type=str,
# #         default=None,
# #         help='Path to YOLO weights'
# #     )
    
# #     parser.add_argument(
# #         '--relationship-weights',
# #         type=str,
# #         default=None,
# #         help='Path to relationship model weights'
# #     )
    
# #     parser.add_argument(
# #         '--caption-model',
# #         type=str,
# #         default=None,
# #         help='Path to caption model directory'
# #     )
    
# #     parser.add_argument(
# #         '--output',
# #         type=str,
# #         default=None,
# #         help='Output file for results (JSON)'
# #     )
    
# #     args = parser.parse_args()
    
# #     # Initialize pipeline
# #     pipeline = RASCInference(
# #         config_path=args.config,
# #         yolo_weights=args.yolo_weights,
# #         relationship_weights=args.relationship_weights,
# #         caption_model=args.caption_model
# #     )
    
# #     # Run inference
# #     results = pipeline.run(args.image)
    
# #     # Save results if requested
# #     if args.output:
# #         import json
# #         with open(args.output, 'w') as f:
# #             json.dump(results, f, indent=2)
# #         logger.info(f"Results saved to {args.output}")


# # if __name__ == "__main__":
# #     main()
