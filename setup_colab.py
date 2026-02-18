"""
Setup script for Google Colab environment
Run this first in your Colab notebook
"""

import os
import sys
from pathlib import Path

def setup_colab_environment():
    """Setup the RASC project in Google Colab"""
    
    print("=" * 60)
    print("Setting up RASC for Google Colab")
    print("=" * 60)
    
    # 1. Install required packages
    print("\n1. Installing dependencies...")
    os.system("pip install -q ultralytics transformers datasets PyYAML pillow tqdm scikit-learn")
    
    # 2. Setup project structure
    print("\n2. Creating project directories...")
    
    base_dir = Path("/content/rasc")
    dirs = [
        "configs",
        "src/data",
        "src/models",
        "src/utils",
        "data/raw/visual_genome",
        "data/processed",
        "data/splits",
        "models/relationship_predictor",
        "models/caption_generator",
        "experiments/runs",
        "logs"
    ]
    
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"   Created {len(dirs)} directories")
    
    # 3. Add to Python path
    print("\n3. Configuring Python path...")
    src_path = str(base_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # 4. Check GPU availability
    print("\n4. Checking GPU availability...")
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        device = "cuda:0"
    else:
        print("   ⚠ No GPU available, using CPU")
        device = "cpu"
    
    # 5. Mount Google Drive (optional)
    print("\n5. Google Drive mounting...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("   ✓ Google Drive mounted at /content/drive")
        drive_mounted = True
    except:
        print("   ⚠ Google Drive not mounted (optional)")
        drive_mounted = False
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nProject directory: {base_dir}")
    print(f"Device: {device}")
    print(f"Google Drive: {'Mounted' if drive_mounted else 'Not mounted'}")
    
    return {
        'base_dir': base_dir,
        'device': device,
        'drive_mounted': drive_mounted
    }


def upload_processed_data():
    """Helper to upload your processed data"""
    from google.colab import files
    
    print("\n" + "=" * 60)
    print("Upload Processed Data")
    print("=" * 60)
    print("\nYou need to upload:")
    print("1. vg_5k_subset.json → data/processed/")
    print("2. label_map.json → data/processed/")
    print("3. relationship_pairs.json → data/processed/relationships/")
    print("4. Train/val/test splits → data/splits/")
    print("\nOptions:")
    print("A. Upload files manually using the file browser (←)")
    print("B. Copy from Google Drive (if mounted)")
    print("C. Use the upload widget below")
    
    choice = input("\nHow would you like to proceed? (A/B/C): ").strip().upper()
    
    if choice == 'C':
        print("\nUpload your files:")
        uploaded = files.upload()
        print(f"\n✓ Uploaded {len(uploaded)} files")
    elif choice == 'B':
        print("\nUse this command to copy from Drive:")
        print("!cp -r /content/drive/MyDrive/rasc_data/* /content/rasc/data/")
    else:
        print("\nUse the file browser on the left to upload files manually")


def create_colab_config(device="cuda:0"):
    """Create a Colab-optimized configuration"""
    
    config_yaml = f"""# RASC Colab Configuration

project:
  name: "rasc"
  version: "1.0.0"
  seed: 42

paths:
  raw_data: "/content/rasc/data/raw/visual_genome"
  processed_data: "/content/rasc/data/processed"
  vg_images: "/content/rasc/data/raw/visual_genome/VG_100K"
  models: "/content/rasc/models"
  experiments: "/content/rasc/experiments/runs"
  logs: "/content/rasc/logs"
  vg_subset: "/content/rasc/data/processed/vg_5k_subset.json"
  label_map: "/content/rasc/data/processed/label_map.json"
  relationship_pairs: "/content/rasc/data/processed/relationships/relationship_pairs.json"
  t5_data: "/content/rasc/data/processed/scene_graphs/t5_data.json"
  splits: "/content/rasc/data/splits"
  detection_data: "/content/rasc/data/processed/detection"

dataset:
  target_images: 5000
  top_k_objects: 150
  spatial_relations:
    - "left of"
    - "right of"
    - "in front of"
    - "behind"
    - "on top of"
    - "under"
    - "inside"
    - "around"
    - "over"
    - "next to"
  splits:
    train: 0.7
    val: 0.15
    test: 0.15

# Colab-optimized settings
detection:
  model_name: "yolov8n"
  pretrained: true
  training:
    epochs: 30  # Reduced for Colab
    batch_size: 16
    image_size: [512, 640]
    workers: 2  # Reduced for Colab
    device: "{device}"
  optimizer:
    name: "AdamW"
    lr0: 0.01
  inference:
    conf_threshold: 0.25
    iou_threshold: 0.45
    max_detections: 15

relationship:
  model_type: "neural_motifs"
  num_classes: 150
  num_relations: 10
  embedding_dim: 128
  training:
    epochs: 15
    batch_size: 64
    learning_rate: 0.001
    weight_decay: 0.0001
  architecture:
    hidden_dim: 512
    dropout: 0.1
    bidirectional: true

captioning:
  model_name: "t5-small"
  training:
    epochs: 5
    batch_size: 8
    learning_rate: 0.0001
    warmup_steps: 500
    eval_steps: 500
    save_steps: 1000
  tokenizer:
    max_input_length: 512
    max_target_length: 64
    padding: "max_length"
    truncation: true
  generation:
    max_length: 64
    num_beams: 4
    early_stopping: true

logging:
  level: "INFO"
  console: true
  file: true
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  tensorboard:
    enabled: false  # Disable for Colab (use W&B instead)

experiment:
  track_metrics: true
  save_checkpoints: true
  checkpoint_frequency: 5
  keep_best_n: 2  # Reduced for Colab storage
"""
    
    config_path = "/content/rasc/configs/config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_yaml)
    
    print(f"✓ Created Colab config at {config_path}")
    return config_path


if __name__ == "__main__":
    # Run setup
    info = setup_colab_environment()
    
    # Create config
    create_colab_config(info['device'])
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Upload your processed data files")
    print("2. Run training scripts")
    print("3. Perform inference")
    print("\nExample:")
    print("  !python /content/rasc/src/models/train_yolo.py --config /content/rasc/configs/config.yaml")
