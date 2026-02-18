# RASC Deployment Checklist

## Pre-Deployment Setup

### 1. Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed from `requirements.txt`
- [ ] GPU drivers installed (if using CUDA)

### 2. Data Preparation
- [ ] Visual Genome dataset downloaded
- [ ] Files placed in `data/raw/visual_genome/`:
  - [ ] `relationships.json`
  - [ ] `objects.json`
  - [ ] `VG_100K/` image directory
- [ ] Verify file paths in `configs/config.yaml`

### 3. Configuration Review
- [ ] Review `configs/config.yaml`
- [ ] Adjust paths for your system
- [ ] Set appropriate batch sizes for your hardware
- [ ] Configure device (`cpu` or `cuda:0`)
- [ ] Set random seed for reproducibility

## Data Processing Pipeline

### Phase 1: Dataset Preparation
- [ ] Run `filter_vg_subset.py` to create 5K subset
- [ ] Run `build_label_map.py` to create object class mapping
- [ ] Run `convert_to_yolo.py` to convert to YOLO format
- [ ] Run `create_splits.py` to create train/val/test splits
- [ ] Verify files in `data/processed/` and `data/splits/`

## Model Training Pipeline

### Phase 2: Object Detection
- [ ] Review YOLO configuration in `configs/yolo.yaml`
- [ ] Run `train_yolo.py` to train object detector
- [ ] Monitor training logs in `logs/`
- [ ] Check experiment outputs in `experiments/runs/`
- [ ] Verify model checkpoints created
- [ ] Note best model path for inference

### Phase 3: Relationship Prediction
- [ ] Run `build_relationship_pairs.py` to create relationship data
- [ ] Verify `data/processed/relationships/relationship_pairs.json`
- [ ] Run `train_relationship.py` to train relationship model
- [ ] Monitor validation accuracy
- [ ] Check for overfitting
- [ ] Save best model path

### Phase 4: Caption Generation
- [ ] Run `build_t5_dataset.py` to prepare caption data
- [ ] Verify `data/processed/scene_graphs/t5_data.json`
- [ ] Run `train_t5.py` to train caption generator
- [ ] Monitor CIDEr scores
- [ ] Save best model path

## Testing & Validation

### Inference Testing
- [ ] Run inference on sample images
- [ ] Verify all three stages work end-to-end
- [ ] Check caption quality
- [ ] Measure inference speed
- [ ] Test with different image sizes

### Performance Evaluation
- [ ] Evaluate object detection (mAP scores)
- [ ] Evaluate relationship prediction (F1, accuracy)
- [ ] Evaluate caption generation (CIDEr, BLEU)
- [ ] Document baseline metrics

## Production Readiness

### Code Quality
- [ ] All imports working correctly
- [ ] No hardcoded paths
- [ ] Error handling in place
- [ ] Logging configured properly
- [ ] Configuration files complete

### Documentation
- [ ] README.md complete and accurate
- [ ] QUICKSTART.md tested and working
- [ ] Code comments and docstrings present
- [ ] Configuration options documented
- [ ] Example commands provided

### Reproducibility
- [ ] Random seeds set
- [ ] Configuration saved with experiments
- [ ] Requirements.txt includes all dependencies
- [ ] Python version documented
- [ ] Hardware requirements noted

## Optional Enhancements

### Advanced Features
- [ ] Set up TensorBoard logging
- [ ] Configure W&B integration (if using)
- [ ] Add batch inference script
- [ ] Create visualization notebooks
- [ ] Add evaluation scripts

### Performance Optimization
- [ ] Profile inference speed
- [ ] Optimize model loading
- [ ] Consider model quantization
- [ ] Test on different hardware
- [ ] Benchmark against baselines

### User Experience
- [ ] Add progress bars to long-running operations
- [ ] Improve error messages
- [ ] Add validation checks
- [ ] Create demo notebook
- [ ] Add example outputs

## Deployment Steps

### 1. Clean Deployment
```bash
# Clone repository
git clone <your-repo-url>
cd rasc-project

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

### 2. Quick Validation
```bash
# Test configuration loading
python -c "from src.utils.config import get_config; c = get_config(); print('Config OK')"

# Test logging
python -c "from src.utils.logger import get_logger; l = get_logger('test'); l.info('Logger OK')"

# Test imports
python -c "from src.models.relationship_models import create_model; print('Models OK')"
```

### 3. Run Pipeline
```bash
# Option 1: Full pipeline
python run_pipeline.py --stage all

# Option 2: Step by step
python run_pipeline.py --stage data
python run_pipeline.py --stage detection
python run_pipeline.py --stage relationships
python run_pipeline.py --stage captions
```

### 4. Test Inference
```bash
# Run on a test image
python src/models/inference.py \
  --image path/to/test/image.jpg \
  --config configs/config.yaml
```

## Troubleshooting

### Common Issues

#### Import Errors
- Check Python path includes project root
- Verify virtual environment is activated
- Reinstall requirements

#### Data Not Found
- Check paths in `config.yaml`
- Verify Visual Genome files downloaded
- Check file permissions

#### CUDA Errors
- Verify CUDA installation
- Check GPU availability
- Try CPU mode first
- Reduce batch sizes

#### Out of Memory
- Reduce batch size in config
- Use smaller model variants
- Enable gradient checkpointing
- Use mixed precision training

### Debug Mode
```bash
# Run with verbose logging
python run_pipeline.py --stage all --log-level DEBUG

# Check specific logs
tail -f logs/[module-name]_*.log
```

## Post-Deployment

### Monitoring
- [ ] Set up log monitoring
- [ ] Track experiment metrics
- [ ] Monitor model performance
- [ ] Check resource usage

### Maintenance
- [ ] Schedule regular model updates
- [ ] Update documentation
- [ ] Track issues and bugs
- [ ] Plan improvements

### Scaling
- [ ] Consider distributed training
- [ ] Optimize inference pipeline
- [ ] Add model caching
- [ ] Implement API endpoints

## Success Criteria

- ✅ All data processing scripts run without errors
- ✅ All models train and converge
- ✅ Inference produces reasonable captions
- ✅ Experiments are tracked and reproducible
- ✅ Code is well-documented
- ✅ Performance meets baseline expectations

## Contact & Support

If you encounter issues:
1. Check the logs in `logs/`
2. Review `PROJECT_SUMMARY.md`
3. Consult `QUICKSTART.md`
4. Open an issue on GitHub

---

**Last Updated**: February 10, 2024
**Version**: 1.0.0
