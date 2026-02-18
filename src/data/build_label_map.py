"""
Build label map from Visual Genome object names
Creates a mapping from object names to class indices
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


class LabelMapBuilder:
    """Build label map for object detection"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize builder with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        self.input_file = Path(self.config.get('paths.vg_subset'))
        self.output_file = Path(self.config.get('paths.label_map'))
        self.top_k = self.config.get('dataset.top_k_objects', 150)
        
        logger.info(f"Building label map from {self.input_file}")
        logger.info(f"Top-K objects: {self.top_k}")
    
    def count_objects(self) -> Counter:
        """
        Count object occurrences in dataset
        
        Returns:
            Counter of object names
        """
        logger.info("Loading dataset...")
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} images")
        
        counter = Counter()
        total_objects = 0
        
        for item in data:
            for obj in item["objects"]:
                name = obj["names"][0].lower()
                counter[name] += 1
                total_objects += 1
        
        logger.info(f"Found {len(counter)} unique object classes")
        logger.info(f"Total object instances: {total_objects}")
        
        return counter
    
    def build_label_map(self, counter: Counter) -> Dict[str, int]:
        """
        Build label map from counter
        
        Args:
            counter: Object name counter
            
        Returns:
            Label map dictionary
        """
        most_common = counter.most_common(self.top_k)
        label_map = {name: idx for idx, (name, count) in enumerate(most_common)}
        
        logger.info(f"Created label map with {len(label_map)} classes")
        logger.info("Top 10 most common objects:")
        for name, count in most_common[:10]:
            logger.info(f"  {name}: {count} instances")
        
        return label_map
    
    def save_label_map(self, label_map: Dict[str, int]):
        """
        Save label map to file
        
        Args:
            label_map: Label map to save
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            json.dump(label_map, f, indent=2)
        
        logger.info(f"Saved label map to {self.output_file}")
    
    def run(self):
        """Run the complete label map building pipeline"""
        logger.info("="*60)
        logger.info("Building label map")
        logger.info("="*60)
        
        counter = self.count_objects()
        label_map = self.build_label_map(counter)
        self.save_label_map(label_map)
        
        logger.info("="*60)
        logger.info("Label map built successfully")
        logger.info("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build object label map from Visual Genome"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    builder = LabelMapBuilder(args.config)
    builder.run()


if __name__ == "__main__":
    main()
