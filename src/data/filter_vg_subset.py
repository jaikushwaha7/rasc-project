"""
Filter Visual Genome dataset to create a subset with spatial relationships
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class VGSubsetFilter:
    """Filter Visual Genome dataset for spatial relationships"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize filter with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        self.raw_vg_dir = Path(self.config.get('paths.raw_data'))
        self.output_dir = Path(self.config.get('paths.processed_data'))
        self.target_count = self.config.get('dataset.target_images', 5000)
        
        self.spatial_relations: Set[str] = set(
            self.config.get('dataset.spatial_relations', [])
        )
        
        logger.info(f"Initialized VG filter with {len(self.spatial_relations)} spatial relations")
        logger.info(f"Target image count: {self.target_count}")
    
    @staticmethod
    def normalize_predicate(text: str) -> str:
        """Normalize predicate text"""
        return text.lower().strip()
    
    def load_json(self, filename: str) -> List[Dict]:
        """Load JSON file from raw VG directory"""
        filepath = self.raw_vg_dir / filename
        logger.info(f"Loading {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} entries from {filename}")
        return data
    
    def filter_images(self) -> List[Dict]:
        """
        Filter images that contain spatial relationships
        
        Returns:
            List of filtered image data
        """
        logger.info("Loading Visual Genome annotations...")
        relationships = self.load_json("relationships.json")
        objects_data = self.load_json("objects.json")
        
        # Map objects by image_id
        image_to_objects = {}
        for entry in objects_data:
            image_id = entry["image_id"]
            image_to_objects[image_id] = entry["objects"]
        
        logger.info(f"Mapped objects for {len(image_to_objects)} images")
        
        valid_images = []
        spatial_rel_counts = {rel: 0 for rel in self.spatial_relations}
        
        logger.info("Filtering images by spatial relationships...")
        
        for entry in tqdm(relationships, desc="Processing images"):
            image_id = entry["image_id"]
            rels = entry["relationships"]
            
            if image_id not in image_to_objects:
                continue
            
            filtered_rels = []
            for r in rels:
                predicate = self.normalize_predicate(r["predicate"])
                
                if predicate in self.spatial_relations:
                    filtered_rels.append(r)
                    spatial_rel_counts[predicate] += 1
            
            if len(filtered_rels) > 0:
                valid_images.append({
                    "image_id": image_id,
                    "relationships": filtered_rels,
                    "objects": image_to_objects[image_id]
                })
        
        logger.info(f"Found {len(valid_images)} valid images")
        logger.info("Spatial relationship distribution:")
        for rel, count in sorted(spatial_rel_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {rel}: {count}")
        
        return valid_images
    
    def create_subset(self, valid_images: List[Dict]) -> List[Dict]:
        """
        Create random subset of target size
        
        Args:
            valid_images: List of valid images
            
        Returns:
            Subset of images
        """
        random.seed(self.config.get('project.seed', 42))
        random.shuffle(valid_images)
        
        subset = valid_images[:self.target_count]
        logger.info(f"Created subset of {len(subset)} images")
        
        return subset
    
    def save_subset(self, subset: List[Dict]):
        """
        Save subset to file
        
        Args:
            subset: Image subset to save
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "vg_5k_subset.json"
        
        with open(output_path, 'w') as f:
            json.dump(subset, f)
        
        logger.info(f"Saved {len(subset)} images to {output_path}")
    
    def run(self):
        """Run the complete filtering pipeline"""
        logger.info("="*60)
        logger.info("Starting Visual Genome subset filtering")
        logger.info("="*60)
        
        valid_images = self.filter_images()
        subset = self.create_subset(valid_images)
        self.save_subset(subset)
        
        logger.info("="*60)
        logger.info("Filtering completed successfully")
        logger.info("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter Visual Genome dataset for spatial relationships"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    filter_pipeline = VGSubsetFilter(args.config)
    filter_pipeline.run()


if __name__ == "__main__":
    main()
