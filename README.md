# RASC: Relationship-Aware Scene Captioning for Accessibility

---

## 🛠 Tech Stack

🐍 Python 3.8+   |   🔥 PyTorch 2.0+
![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 🧠 Model Architecture

![Object Detection](https://img.shields.io/badge/Object_Detection-YOLOv8-blueviolet?style=for-the-badge)
![Relationship Model](https://img.shields.io/badge/Relationship_Model-MLP-orange?style=for-the-badge)
![Scene Graph](https://img.shields.io/badge/Scene_Graph-Pairwise_Relations-9cf?style=for-the-badge)
![Caption Generator](https://img.shields.io/badge/Caption_Model-T5_Scene-green?style=for-the-badge)
![Pipeline](https://img.shields.io/badge/Pipeline-Detection→Relation→Caption-black?style=for-the-badge)

---

## 📂 Dataset

![COCO](https://img.shields.io/badge/Dataset-COCO-FF9900?style=for-the-badge)
![Custom Dataset](https://img.shields.io/badge/Custom-Spatial_Relations-blue?style=for-the-badge)
![Synthetic Pairs](https://img.shields.io/badge/Training-Synthetic_Pairs-lightgrey?style=for-the-badge)

---

## 🔁 CI/CD

![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code_Style-PEP8-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)



## 📌 Overview

RASC is a computer vision and vision-language project that generates **relationship-aware scene descriptions** to improve accessibility for visually impaired users. Instead of listing detected objects, the system explicitly models **spatial relationships** (e.g., *left of*, *on*, *near*) and produces cognitively meaningful natural language descriptions.

**Example Output:**
> *"A person walking a dog on the left side of a tree-lined street."*

## 🎯 Motivation

Most existing accessibility tools provide flat object lists or generic captions that miss spatial and relational context. This project addresses that gap by:

- Explicitly learning object relationships
- Converting structured scene graphs into natural language
- Supporting better scene understanding and navigation for visually impaired users

## 🧠 Method Overview

RASC uses a **three-stage pipeline**:

```
Image → YOLOv8 → Scene Graph (Objects + Relationships) → T5 → Caption
```

### 1. Object Detection
- **Model**: YOLOv8n (nano/small)
- **Training**: Pretrained on COCO, optionally fine-tuned on Visual Genome
- **Output**: Object bounding boxes and class predictions

### 2. Relationship Prediction
- **Architectures**: 
  - Simple MLP baseline
  - Neural Motifs (Context-aware LSTM-based approach)
- **Input**: Object pairs with bounding boxes
- **Output**: Spatial relationship predictions

### 3. Scene Graph → Natural Language
- **Model**: T5-small
- **Training**: Fine-tuned to convert structured scene graphs to captions
- **Output**: Natural language scene descriptions

## 📊 Dataset

- **Source**: [Visual Genome](https://visualgenome.org/) (Krishna et al., 2017)
- **Subset**: ~5,000 images with spatial relationships
- **Object Classes**: Top 150 most frequent
- **Relationships**: 10 spatial predicates
  - left of, right of, in front of, behind
  - on top of, under, inside, around, over, next to

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Cluster Usage
- Directory: **cluster/**
- Setup & details: **cluster/readme.md**

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rasc.git
cd rasc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Visual Genome dataset
# Follow instructions at: https://visualgenome.org/api/v0/api_home.html
```

## 📁 Project Structure

```
rasc-project/
├── configs/
│   ├── config.yaml              # Main configuration
│   └── yolo.yaml                # YOLO dataset config
├── src/
│   ├── data/                    # Data processing
│   │   ├── filter_vg_subset.py
│   │   ├── build_label_map.py
│   │   ├── convert_to_yolo.py
│   │   └── create_splits.py
│   ├── models/                  # Model training
│   │   ├── train_yolo.py
│   │   ├── train_relationship.py
│   │   ├── train_t5.py
│   │   ├── relationship_models.py
│   │   └── inference.py
│   └── utils/                   # Utilities
│       ├── config.py
│       ├── logger.py
│       └── experiment.py
│  
│  
├── experiments/                 # Experiment outputs
│   └── runs/
├── logs/                        # Log files
├── data/                        # Data directory
│   ├── raw/
│   ├── processed/
│   └── splits/
│   └── download_vg.sh
├──outputs
│   └── eval_relationships/      # Evalation results
├── models/                      # Saved models
├── requirements.txt
├── run_pipeline.py 
└── README.md
```

## 🔧 Usage

### Step 1: Data Preparation
#### skip to step 2 as data is already present in the data folder

```bash
# Filter Visual Genome subset
python src/data/filter_vg_subset.py --config configs/config.yaml

# Build label map
python src/data/build_label_map.py --config configs/config.yaml

# Convert to YOLO format
python src/data/convert_to_yolo.py --config configs/config.yaml

# Create train/val/test splits
python src/data/create_splits.py --config configs/config.yaml
```

### Step 2: Model Training

```bash
# Train object detector
python src/models/train_yolo.py \
  --config configs/config.yaml \
  --experiment-name yolo_experiment_1

# Build relationship pairs
python src/data/build_relationship_pairs.py --config configs/config.yaml

# Train relationship model
python src/models/train_relationship.py \
  --config configs/config.yaml \
  --experiment-name relationship_neural_motifs

# Prepare caption data
python src/data/build_t5_dataset.py --config configs/config.yaml

# Train caption generator
python src/models/train_t5.py \
  --config configs/config.yaml \
  --experiment-name t5_captioning
or 
 python -m src.models.train_t5.py  --config configs/config.yaml --experiment-name t5_captioning

```

### Step 3: Inference

```bash
# Run inference on a single image
python src/models/inference.py  --image data/test/alexanderplatz.png   --config configs/config.yaml
```

## App run
```bash 
streamlit run rasc_demo_app5.py
```

## 📊 Evaluation

The system is evaluated at each stage:

| Component | Metrics |
|-----------|---------|
| **Object Detection** | mAP@0.5, mAP@0.5:0.95, Precision, Recall |
| **Relationship Prediction** | F1-Score, Accuracy, Confusion Matrix |
| **Caption Generation** | CIDEr, BLEU-4, METEOR, ROUGE-L |

### Running Evaluation

```bash
# Evaluate object detection
python src/models/train_yolo.py --eval-only --weights path/to/best.pt

# Evaluate relationship prediction
 python -m src.models.evaluate_relationships    --config configs/config.yaml     --model-path models/relationship_predictor/neural_motifs_best.pt     --data-path data/processed/relationships/test/    --batch-size 64     --output-dir outputs/eval_relationships    


# Evaluate captions
python src/models/evaluate_captions.py --config configs/config.yaml
```

## 🔬 Experiments and Logging

The project includes comprehensive experiment tracking:

- **Configuration Management**: YAML-based configs for all hyperparameters
- **Logging**: Structured logging to console and files
- **Metrics Tracking**: Automatic logging of training metrics
- **Checkpointing**: Automatic model checkpoint saving
- **Experiment Reproducibility**: All experiments saved with configs and metrics

### Viewing Experiment Results

```bash
# List all experiments
ls experiments/runs/

# View metrics for a specific experiment
cat experiments/runs/yolo_detection_20240210_143022/metrics/metrics.json
```

## 🧪 Ablation Studies

The codebase supports various ablation studies:

1. **Pretrained vs Fine-tuned Detector**
   - Compare COCO-pretrained vs VG fine-tuned YOLO
   
2. **Relationship Models**
   - MLP baseline vs Neural Motifs
   
3. **Caption Generation**
   - Different T5 model sizes (small, base, large)
   - Different scene graph serialization strategies

## ⚠️ Limitations

- Limited dataset subset (5K images from Visual Genome)
- Only spatial relationships (no temporal or causal reasoning)
- No user study with visually impaired participants
- Multi-stage pipeline may propagate errors
- Computational requirements for full pipeline



## 👥 Team & Contributions

- **Jai Kushwaha** – Object detection, relationship modeling, evaluation, Caption generation, poster design, presentation
- **Caner Gel** – Code testing, poster design, presentation

## 📚 References

1. Krishna et al., *Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations*, IJCV 2017
2. Zellers et al., *Neural Motifs: Scene Graph Parsing with Global Context*, CVPR 2018
3. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 2020
4. Ultralytics, *YOLOv8*, 2023
5. DSGG:(CVPR 2024 paper): Dense Relation Transformer for an End-to-end Scene Graph Generation: Zeeshan Hayder, Xuming He
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


