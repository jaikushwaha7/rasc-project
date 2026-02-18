# RASC Colab Quick Start Guide

## 🚀 Running RASC on Google Colab

Since you already have processed data locally, here's how to quickly get started with training on Colab.

## Option 1: Using the Notebook (Recommended)

### Step 1: Upload to Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/RASC_Colab_Training.ipynb`
3. Enable GPU: **Runtime → Change runtime type → GPU → T4**

### Step 2: Upload Your Processed Data

Choose one of these methods:

**Method A: Google Drive (Recommended for large datasets)**
```python
# In Colab:
from google.colab import drive
drive.mount('/content/drive')

# Upload your data to Drive first, then copy:
!cp -r /content/drive/MyDrive/rasc_data/* /content/rasc/data/
```

**Method B: Direct Upload**
```python
# In Colab:
from google.colab import files
uploaded = files.upload()  # Upload your processed files
```

### Step 3: Run the Notebook
Just execute the cells in order! The notebook handles:
- ✅ Environment setup
- ✅ Data verification
- ✅ Model training (all 3 stages)
- ✅ Inference
- ✅ Saving results to Drive

---

## Option 2: Manual Setup (Command Line)

If you prefer command-line execution:

### 1. Initial Setup

```python
# In a new Colab cell:

# Install dependencies
!pip install -q ultralytics transformers datasets PyYAML pillow tqdm scikit-learn

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone/upload project
!unzip /content/rasc-project.zip -d /content/

# Setup paths
import sys
sys.path.insert(0, '/content/rasc-project/src')
```

### 2. Upload Processed Data

Create a folder structure on Google Drive:
```
MyDrive/
└── rasc_data/
    ├── vg_5k_subset.json
    ├── label_map.json
    ├── relationship_pairs.json
    └── splits/
        ├── train/
        ├── val/
        └── test/
```

Then in Colab:
```bash
!mkdir -p /content/rasc-project/data/processed/relationships
!mkdir -p /content/rasc-project/data/splits

!cp /content/drive/MyDrive/rasc_data/vg_5k_subset.json \
   /content/rasc-project/data/processed/

!cp /content/drive/MyDrive/rasc_data/label_map.json \
   /content/rasc-project/data/processed/

!cp /content/drive/MyDrive/rasc_data/relationship_pairs.json \
   /content/rasc-project/data/processed/relationships/

!cp -r /content/drive/MyDrive/rasc_data/splits/* \
   /content/rasc-project/data/splits/
```

### 3. Create Colab Config

```python
import yaml
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

config = {
    'paths': {
        'processed_data': '/content/rasc-project/data/processed',
        'models': '/content/rasc-project/models',
        'experiments': '/content/rasc-project/experiments/runs',
        'label_map': '/content/rasc-project/data/processed/label_map.json',
        'relationship_pairs': '/content/rasc-project/data/processed/relationships/relationship_pairs.json',
        'splits': '/content/rasc-project/data/splits'
    },
    'detection': {
        'model_name': 'yolov8n',
        'training': {
            'epochs': 30,
            'batch_size': 16,
            'device': device
        }
    },
    'relationship': {
        'model_type': 'neural_motifs',
        'num_classes': 150,
        'num_relations': 10,
        'training': {'epochs': 15, 'batch_size': 64}
    },
    'captioning': {
        'model_name': 't5-small',
        'training': {'epochs': 5, 'batch_size': 8}
    }
}

with open('/content/rasc-project/configs/config.yaml', 'w') as f:
    yaml.dump(config, f)
```

### 4. Train Models

```bash
# Stage 1: Object Detection
%cd /content/rasc-project
!python src/models/train_yolo.py \
  --config configs/config.yaml \
  --experiment-name yolo_colab

# Stage 2: Relationship Prediction
!python src/models/train_relationship.py \
  --config configs/config.yaml \
  --experiment-name rel_colab

# Stage 3: Caption Generation
# First prepare data
!python src/data/build_t5_dataset.py --config configs/config.yaml

# Then train
!python src/models/train_t5.py \
  --config configs/config.yaml \
  --experiment-name t5_colab
```

### 5. Run Inference

```python
# Upload test image
from google.colab import files
uploaded = files.upload()
test_img = list(uploaded.keys())[0]

# Run inference
!python src/models/inference.py \
  --image {test_img} \
  --config configs/config.yaml
```

---

## 💡 Tips for Colab

### Memory Management
```python
# Clear GPU memory between stages
import torch
torch.cuda.empty_cache()

# Monitor GPU usage
!nvidia-smi
```

### Save Frequently
```bash
# Save models to Drive after each stage
!cp -r /content/rasc-project/models/* \
   /content/drive/MyDrive/rasc_models/

!cp -r /content/rasc-project/experiments/* \
   /content/drive/MyDrive/rasc_experiments/
```

### Resume Training
```python
# If disconnected, reload from Drive
!cp -r /content/drive/MyDrive/rasc_models/* \
   /content/rasc-project/models/

!cp -r /content/drive/MyDrive/rasc_experiments/* \
   /content/rasc-project/experiments/
```

---

## ⚠️ Common Issues

### 1. Runtime Disconnected
**Solution**: Save checkpoints to Drive regularly
```python
# Add this after each training stage
!cp -r /content/rasc-project/experiments/runs/* \
   /content/drive/MyDrive/rasc_checkpoints/
```

### 2. Out of Memory
**Solution**: Reduce batch sizes in config
```yaml
detection:
  training:
    batch_size: 8  # Instead of 16

relationship:
  training:
    batch_size: 32  # Instead of 64
```

### 3. Slow Data Upload
**Solution**: Use Google Drive instead of direct upload
- Upload data to Drive using desktop app (faster)
- Then copy from Drive in Colab

### 4. Session Timeout
**Solution**: Use Colab Pro or run stages separately
```python
# Train one stage per session
# Save everything to Drive
# Start new session for next stage
```

---

## 📊 Monitoring Training

### View Live Logs
```bash
# In a separate cell while training
!tail -f /content/rasc-project/logs/*.log
```

### Check Metrics
```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('/content/rasc-project/experiments/runs/[exp_name]/metrics/metrics.json') as f:
    metrics = json.load(f)

# Plot
if 'train_loss' in metrics:
    losses = [m['value'] for m in metrics['train_loss']]
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
```

---

## 🎯 Expected Training Times (T4 GPU)

| Stage | Time |
|-------|------|
| Object Detection (30 epochs) | ~40-60 min |
| Relationship Prediction (15 epochs) | ~15-20 min |
| Caption Generation (5 epochs) | ~20-30 min |
| **Total** | **~75-110 min** |

---

## ✅ Verification Checklist

Before starting:
- [ ] GPU enabled (T4 recommended)
- [ ] Data uploaded to Drive or Colab
- [ ] Config file created
- [ ] Python path configured
- [ ] Dependencies installed

After training:
- [ ] Models saved to Drive
- [ ] Metrics logged
- [ ] Inference tested
- [ ] Results backed up

---

## 🆘 Need Help?

1. **Check logs**: `/content/rasc-project/logs/`
2. **Verify data**: `!ls -lh /content/rasc-project/data/`
3. **GPU status**: `!nvidia-smi`
4. **Free memory**: `torch.cuda.empty_cache()`

---

**Happy Training! 🚀**
