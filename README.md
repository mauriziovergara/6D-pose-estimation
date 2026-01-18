# 6D Pose Estimation with PoseNet and YOLO

A deep learning pipeline for 6D pose estimation on the LineMOD dataset using YOLO for object detection and a custom PoseNet architecture for pose regression.

## Project Overview

This project implements a complete pipeline for 6D object pose estimation:
- **Object Detection**: YOLOv8 fine-tuned on LineMOD dataset for bounding box prediction
- **Pose Estimation**: Custom PoseNet architecture (ResNet50 backbone) for rotation and translation prediction
- **Evaluation Metrics**: ADD (Average Distance of Model Points) and ADD-S (symmetric objects) metrics

### Key Features
- 🎯 Multi-class object detection with YOLO
- 📐 Geometric-aware pose regression using pinhole camera model
- 🔄 Quaternion-based rotation representation
- 🎨 Automatic loss weighting for multi-task learning
- 📊 Comprehensive evaluation with visualization support
- 💾 Checkpoint management and model serialization

## Dataset

The project uses the **LineMOD dataset**, a benchmark dataset for 6D pose estimation containing:
- 13 object classes
- ~1000 images per object
- RGB-D data with camera calibration matrices
- Ground truth poses (rotation + translation)

### LineMOD Structure
```
Linemod_preprocessed/
├── data/
│   ├── 01/  (object 1)
│   ├── 02/  (object 2)
│   └── ...  (objects 1-15)
│       ├── gt.yml          # Ground truth poses
│       ├── info.yml        # Camera parameters
│       ├── train.txt       # Training split
│       ├── test.txt        # Test split
│       ├── rgb/            # RGB images
│       ├── depth/          # Depth maps
│       └── mask/           # Instance masks
└── models/
    └── models_info.yml     # 3D model information
```

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM recommended

### Setup

1. **Clone repository**
```bash
git clone https://github.com/mauriziovergara/6D-pose-estimation
cd 6d-pose-estimation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download LineMOD dataset**
- Download from [BOP Challenge](https://bop.felk.cvut.cz/)
- Extract to `Linemod_preprocessed/` directory
- Verify structure matches the layout above

## Project Structure

```
6D_pinhole/
├── Code/
│   ├── dataset.py                      # LineMOD dataset loader
│   ├── pose_net_rgb_geometric.py       # PoseNet model architecture
│   ├── train_posenet_rgb.py            # Training script
│   ├── train_yolo_linemod.py           # YOLO fine-tuning
│   ├── prepare_data.py                 # Data preprocessing
│   ├── run_yolo_inference.py           # YOLO inference
│   ├── add_posenet.py                  # ADD metric evaluation
│   ├── adds_posenet.py                 # ADD-S metric (symmetric objects)
│   ├── visualize_samples.py            # Visualization tools
│   └── requirements.txt                # Dependencies
├── Pose_weight/                        # Trained pose model checkpoints
├── Predictions/                        # YOLO predictions & pose outputs
└── README.md                           # This file
```

## Usage

### 1. Prepare Data

Convert LineMOD to YOLO format:
```bash
cd Code
python prepare_data.py
```

### 2. Train YOLO Detector

Fine-tune YOLOv8 on LineMOD:
```bash
python train_yolo_linemod.py \
    --data ../data/linemod_multi.yaml \
    --epochs 20 \
    --batch 16 \
    --device cuda
```

### 3. Run YOLO Inference

Generate bounding box predictions:
```bash
python run_yolo_inference.py \
    --yolo_weights yolo_linemod_runs/finetune_all_objs/weights/best.pt \
    --dataset_root ../Linemod_preprocessed/Linemod_preprocessed \
    --output yolo_predictions.yaml \
    --conf 0.25
```

### 4. Train Pose Estimator

Train PoseNet for 6D pose:
```bash
python train_posenet_rgb.py \
    --root_dir ../Linemod_preprocessed/Linemod_preprocessed/data \
    --models_dir ../Linemod_preprocessed/Linemod_preprocessed/models \
    --pred_boxes yolo_predictions.yaml \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda
```

### 5. Evaluate Results

**ADD Metric (all objects):**
```bash
python add_posenet.py \
    --root_dir ../Linemod_preprocessed/Linemod_preprocessed/data \
    --models_dir ../Linemod_preprocessed/Linemod_preprocessed/models \
    --checkpoint Pose_weight/best_pose.pt \
    --pred_boxes Predictions/yolo_predictions.yaml \
    --output_csv results_add.csv
```

**ADD-S Metric (symmetric objects):**
```bash
python adds_posenet.py \
    --root_dir ../Linemod_preprocessed/Linemod_preprocessed/data \
    --models_dir ../Linemod_preprocessed/Linemod_preprocessed/models \
    --checkpoint Pose_weight/best_pose.pt \
    --pred_boxes Predictions/yolo_predictions.yaml \
    --output_csv results_adds.csv \
    --symmetric_only
```

### 6. Visualize Results

Generate visualization with predicted poses:
```bash
python visualize_samples.py \
    --root_dir ../Linemod_preprocessed/Linemod_preprocessed/data \
    --yolo_pred_path Predictions/yolo_predictions.yaml \
    --pose_pred_path Predictions/best_pose_predictions.yaml \
    --output_dir visualizations/
```

## Model Architecture

### PoseNetRGBGeometric

```
RGB Input (224×224×3)
    ↓
[ResNet50 Backbone] ──→ Rotation Head → Quaternion (4D)
    ↓
[Lightweight CNN] ──→ Z-Depth Predictor → Z (1D)
    ↓
[Pinhole Camera Model] → X, Y from Z and bbox center
    ↓
Output: (Rotation: quat, Translation: [X, Y, Z])
```

**Key Components:**
- **Rotation Head**: ResNet50 features → Quaternion representation
- **Z-Depth CNN**: Dedicated branch for depth prediction
- **Geometric Layer**: Pinhole camera model for X, Y recovery using camera intrinsics

## Loss Function

Multi-task learning with automatic weight balancing (Kendall et al.):

$$L = \frac{1}{2\sigma_{rot}^2} L_{rot} + \frac{1}{2\sigma_{trans}^2} L_{trans} + \log(\sigma_{rot}) + \log(\sigma_{trans})$$

Where:
- $L_{rot}$ = Geodesic distance on SO(3) manifold (quaternion loss)
- $L_{trans}$ = L2 distance for translation
- $\sigma_{rot}, \sigma_{trans}$ = Learned uncertainty weights

## Evaluation Metrics

### ADD (Average Distance of Model Points)
- Success if: mean point distance < 0.1 × object diameter
- Formula: $ADD = \frac{1}{N}\sum_{i=1}^{N} ||R_{pred}p_i + t_{pred} - (R_{gt}p_i + t_{gt})||$

### ADD-S (Symmetric Objects)
- Special handling for symmetric objects (eggbox, glue)
- Uses minimum distance after considering symmetries

## Configuration Parameters

### Training (`train_posenet_rgb.py`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--lambda_rot`: Rotation loss weight (default: 1.0)
- `--lambda_trans`: Translation loss weight (default: 1.0)
- `--rotation_loss_type`: "chordal" or "geodesic" (default: "chordal")
- `--freeze_backbone_ratio_val`: Backbone freeze ratio (0.0-1.0)

### Evaluation
- `--val_split`: Validation set fraction (default: 0.2)
- `--add_thresh`: ADD threshold multiplier (default: 0.02 = 0.1×diameter)

## Results

Example results on LineMOD (validation set):

| Class | ADD@0.1d (%) | ADD Mean (cm) | Trans Err (cm) | Rot Err (°) |
|-------|--------------|---------------|----------------|------------|
| 01    | 85.2         | 2.3           | 1.1            | 5.2        |
| 02    | 78.5         | 3.1           | 1.8            | 7.3        |
| ...   | ...          | ...           | ...            | ...        |
| Global| 82.1         | 2.6           | 1.4            | 6.1        |

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` 
- Enable gradient accumulation
- Reduce image resolution

### Low Accuracy
- Verify LineMOD dataset integrity
- Check YOLO bounding box quality
- Increase training epochs
- Adjust loss weights (`--lambda_rot`, `--lambda_trans`)

### Missing Predictions
- Ensure YOLO predictions YAML has correct format
- Verify camera matrix consistency
- Check LineMOD folder structure

## References

- **LineMOD Dataset**: [Link](https://bop.felk.cvut.cz/)
- **YOLO**: Ultralytics YOLOv8 [GitHub](https://github.com/ultralytics/ultralytics)
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition"
- **Quaternions**: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

## License

[Specify your license - MIT, Apache 2.0, etc.]

## Contact

- Author: [Your Name]
- Email: [Your Email]
- Repository: [GitHub URL]

## Acknowledgments

- LineMOD dataset creators and BOP Challenge organizers
- Ultralytics for YOLOv8
- PyTorch team for the deep learning framework
