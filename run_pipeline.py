#!/usr/bin/env python3
"""
Main pipeline script for RASC
Runs the complete pipeline from data processing to inference
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from utils.logger import get_logger

logger = get_logger(__name__)


class RASCPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.src_dir = Path(__file__).parent / 'src'
    
    # def run_script(self, script_path: str, **kwargs):
    #     """Run a Python script with arguments"""
    #     cmd = [sys.executable, str(self.src_dir / script_path)]
    #     cmd.extend(['--config', self.config_path])
        
    #     for key, value in kwargs.items():
    #         if value is not None:
    #             cmd.extend([f'--{key}', str(value)])
        
    #     logger.info(f"Running: {' '.join(cmd)}")
    #     result = subprocess.run(cmd, check=True)
    #     return result.returncode == 0
    def run_script(self, script_path: str, **kwargs):
        """Run a Python script with arguments"""
        cmd = [sys.executable, str(self.src_dir / script_path)]
        cmd.extend(['--config', self.config_path])
        
        for key, value in kwargs.items():
            if value is not None:
                # Convert experiment_name to --experiment-name
                flag = f"--{key.replace('_', '-')}"
                cmd.extend([flag, str(value)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        # Adding capture_output=True can help debug exactly what the sub-script complained about
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    
    def data_preparation(self):
        """Run all data preparation steps"""
        logger.info("="*60)
        logger.info("PHASE 1: Data Preparation")
        logger.info("="*60)
        
        steps = [
            ("Filter VG subset", "data/filter_vg_subset.py"),
            ("Build label map", "data/build_label_map.py"),
            ("Convert to YOLO", "data/convert_to_yolo.py"),
            ("Create splits", "data/create_splits.py"),
        ]
        
        for name, script in steps:
            logger.info(f"Step: {name}")
            self.run_script(script)
    
    def train_detection(self, experiment_name: str = None):
        """Train object detection model"""
        logger.info("="*60)
        logger.info("PHASE 2: Object Detection Training")
        logger.info("="*60)
        
        self.run_script(
            "models/train_yolo.py",
            experiment_name=experiment_name or "yolo_detection"
        )
    
    def prepare_relationships(self):
        """Prepare relationship data"""
        logger.info("="*60)
        logger.info("PHASE 3: Relationship Data Preparation")
        logger.info("="*60)
        
        self.run_script("data/build_relationship_pairs.py")
    
    def train_relationships(self, experiment_name: str = None):
        """Train relationship prediction model"""
        logger.info("="*60)
        logger.info("PHASE 4: Relationship Prediction Training")
        logger.info("="*60)
        
        self.run_script(
            "models/train_relationship.py",
            experiment_name=experiment_name or "relationship_neural_motifs"
        )
    
    def prepare_captions(self):
        """Prepare caption generation data"""
        logger.info("="*60)
        logger.info("PHASE 5: Caption Data Preparation")
        logger.info("="*60)
        
        self.run_script("data/build_t5_dataset.py")
    
    def train_captions(self, experiment_name: str = None):
        """Train caption generation model"""
        logger.info("="*60)
        logger.info("PHASE 6: Caption Generation Training")
        logger.info("="*60)
        
        self.run_script(
            "models/train_t5.py",
            experiment_name=experiment_name or "t5_captioning"
        )
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        logger.info("="*60)
        logger.info("STARTING FULL RASC PIPELINE")
        logger.info("="*60)
        
        try:
            self.data_preparation()
            self.train_detection()
            self.prepare_relationships()
            self.train_relationships()
            self.prepare_captions()
            self.train_captions()
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="RASC Pipeline - Relationship-Aware Scene Captioning"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=[
            'all', 'data', 'detection', 'relationships', 
            'captions', 'inference'
        ],
        default='all',
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Base name for experiments'
    )
    
    args = parser.parse_args()
    
    pipeline = RASCPipeline(args.config)
    
    if args.stage == 'all':
        pipeline.run_full_pipeline()
    elif args.stage == 'data':
        pipeline.data_preparation()
    elif args.stage == 'detection':
        pipeline.train_detection(args.experiment_name)
    elif args.stage == 'relationships':
        pipeline.prepare_relationships()
        pipeline.train_relationships(args.experiment_name)
    elif args.stage == 'captions':
        pipeline.prepare_captions()
        pipeline.train_captions(args.experiment_name)
    else:
        logger.error(f"Unknown stage: {args.stage}")
        sys.exit(1)


if __name__ == "__main__":
    main()
