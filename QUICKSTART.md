# RASC Quick Start Guide

This is the guide that will help get started with RASC quickly.

## Prerequisites

```bash
# Python 3.8 or higher
python --version

# (Optional) CUDA for GPU training
nvidia-smi
```

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/jaikushwaha7/rasc.git
cd rasc-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Visual Genome Dataset

```bash
# Create data directory
mkdir -p data/raw/visual_genome

# Download Visual Genome (visit https://visualgenome.org/)
# You need:
# - relationships.json
# - objects.json
# - VG_100K/ (images folder)

# Place them in data/raw/visual_genome/
```
#### Download complete data
```bash
# Download Visual Genome dataset
data/download_vg.sh
```
## Quick Training Pipeline

### Option 1: Run Everything at Once

```bash
# This will run the complete pipeline
python run_pipeline.py --stage all --config configs/config.yaml
```

### Option 2: Step-by-Step

#### Step 1: Prepare Data (5-10 minutes)

```bash
python run_pipeline.py --stage data
```

This will:
- Filter 5K images with spatial relationships
- Build label map for 150 object classes
- Convert to YOLO format
- Create train/val/test splits

#### Step 2: Train Object Detector (30-60 minutes)

```bash
python run_pipeline.py --stage detection --experiment-name my_yolo_v1
```

#### Step 3: Train Relationship Model (15-30 minutes)

```bash
python run_pipeline.py --stage relationships --experiment-name my_relationships_v1
```

#### Step 4: Train Caption Generator (20-40 minutes)

```bash
python run_pipeline.py --stage captions --experiment-name my_captions_v1
```

## Running Inference

### Single Image

```bash
python src/models/inference.py \
  --image path/to/your/image.jpg \
  --config configs/config.yaml
```

### With Custom Models

```bash
python src/models/inference.py \
  --image path/to/your/image.jpg \
  --yolo-weights experiments/runs/yolo_detection_xxx/weights/best.pt \
  --relationship-weights models/relationship_predictor/neural_motifs_best.pt \
  --caption-model models/caption_generator/t5_scene
```

## Configuration

Edit `configs/config.yaml` to customize:

- Dataset size and splits
- Model architectures
- Training hyperparameters
- Paths and directories

## Common Issues

### 1. CUDA Out of Memory

```yaml
# In configs/config.yaml, reduce batch sizes:
detection:
  training:
    batch_size: 8  # Reduce from 16

relationship:
  training:
    batch_size: 32  # Reduce from 64
```

### 2. Missing Data

Make sure Visual Genome files are in the correct location:
```
data/raw/visual_genome/
├── relationships.json
├── objects.json
└── VG_100K/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

### 3. Import Errors

```bash
# Make sure you're in the project root
cd rasc-project

# And have activated the virtual environment
source venv/bin/activate
```


## Monitoring Training

### View Logs

```bash
# Console logs
tail -f logs/*.log

# Experiment logs
tail -f experiments/runs/your_experiment/logs/*.log
```

### Check Metrics

```bash
# View metrics JSON
cat experiments/runs/your_experiment/metrics/metrics.json

# Or use jq for pretty printing
cat experiments/runs/your_experiment/metrics/metrics.json | jq .
```


## Next Steps

1. **Experiment with different configurations**
   - Try different model sizes (yolov8s, t5-base)
   - Adjust hyperparameters
   - Test different relationship models

2. **Evaluate your models**
   ```bash
   python src/models/evaluate.py --config configs/config.yaml
   ```

3. **Run batch inference**
   ```bash
   python scripts/batch_inference.py --input-dir path/to/images --output-dir results
   ```


## Getting Help

- Check the [README](README.md) for detailed documentation
- Review [Configuration Guide](docs/configuration.md)
- Open an issue on GitHub
- Contact the team

## Tips for Best Results

1. **Use GPU if available** - Training is much faster
2. **Start with default config** - It's been tuned for good results
3. **Monitor validation metrics** - Watch for overfitting
4. **Save your experiments** - Keep track of what works
5. **Iterate quickly** - Test with small datasets first

---

