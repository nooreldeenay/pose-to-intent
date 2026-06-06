# Pose to Intent

Pedestrian crossing prediction using pose, bbox, segmentation, traffic, and vehicle features on the JAAD dataset.

## Overview

This repository contains:

1. **Keypoint Extraction** (`extract_poses.py`): ViTPose-based pose estimation
2. **Segmentation Extraction** (`extract_seg_maps.py`): Mask2Former semantic segmentation
3. **Teacher Model** (`train_teacher.py`): TrafficVehicleTransformer (~430K params, AUC=0.934)
4. **Student Models** (`train_student.py`): Lightweight models (~53K params) with/without KD
5. **Evaluation** (`evaluate_model.py`): Test any trained checkpoint

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- OpenCV
- scikit-learn
- transformers (HuggingFace)
- tqdm

Install dependencies:
```bash
pip install torch numpy opencv-python scikit-learn transformers tqdm
```

## Environment Variables

Set these before running any script:

```bash
# Required: Path to JAAD dataset
export JAAD_PATH=/path/to/JAAD

# Optional: Override default cache directories
export POSE_CACHE_DIR=~/ped_data/jaad/poses
export SEG_CACHE_DIR=~/ped_data/jaad/seg_maps
export SAVE_DIR=~/ped_data/pose_to_intent
```

## Data Preparation

### Step 1: Extract Keypoints

Extract ViTPose keypoints for all frames in the JAAD dataset:

```bash
export JAAD_PATH=/path/to/JAAD
python extract_poses.py
```

This saves 34-D pose vectors (17 keypoints x 2 coordinates) to `$POSE_CACHE_DIR/pose_set01.pkl`.

Options:
- `--regen_data`: Recompute all poses (ignore cache)
- `--splits train val test`: Process specific splits
- `--model usyd-community/vitpose-base-simple`: Use different ViTPose model

### Step 2: Extract Segmentation Maps

Extract Mask2Former semantic segmentation maps for all frames:

```bash
export JAAD_PATH=/path/to/JAAD
python extract_seg_maps.py
```

This saves uint8 seg maps (19 Cityscapes classes) to `$SEG_CACHE_DIR/<vid_id>/<frame>.npy`.

Options:
- `--regen_data`: Recompute all seg maps (ignore cache)
- `--splits train val test`: Process specific splits

### Step 3: Verify Data

After extraction, verify the cache directories exist:
```bash
ls $POSE_CACHE_DIR/pose_set01.pkl
ls $SEG_CACHE_DIR/
```

## Training

### Train Teacher Model

```bash
export JAAD_PATH=/path/to/JAAD
python train_teacher.py
```

Trains the TrafficVehicleTransformer teacher model with SWA (average last 10 epochs).

### Train Student Model

```bash
export JAAD_PATH=/path/to/JAAD

# Without knowledge distillation
python train_student.py --model conv1d --hidden 64

# With output-level KD
python train_student.py --model conv1d --hidden 64 --use_kd

# With representation-level KD
python train_student.py --model conv1d --hidden 64 --use_repr_kd

# With bbox features
python train_student.py --model conv1d --hidden 64 --use_bbox

# With specific activation
python train_student.py --model conv1d --hidden 64 --activation gelu
```

## Evaluation

Evaluate any trained checkpoint:

```bash
export JAAD_PATH=/path/to/JAAD

# Evaluate teacher
python evaluate_model.py --model_path /path/to/teacher/best_model.pt --model_type teacher

# Evaluate student
python evaluate_model.py --model_path /path/to/student/best_model.pt --model_type student_conv1d
```

## Model Architecture

### Teacher (TrafficVehicleTransformer)

```
Bbox Stream:    Linear(14->64) -> Transformer(2 layers, 4 heads) -> mean pooling
Seg Stream:     Embedding(19->16) -> CNN(3 layers) -> Attention pooling -> Linear(64->64)
Pose Stream:    Linear(49->64) -> Conv1D(k=3) + Conv1D(k=7) -> Linear(128->64)
Traffic Stream: Linear(5->64) -> Transformer(2 layers, 4 heads) -> mean pooling
Vehicle Stream: Linear(5->64) -> Transformer(2 layers, 4 heads) -> mean pooling
Classifier:     Linear(64*5->1)
```

### Student (StudentConv1D)

```
Pose Stream:    Linear(49->64) -> Conv1D(k=3) + Conv1D(k=7) -> Linear(128->64)
Classifier:     Linear(64->1)
```

## Feature Engineering

### Pose Features (49-D)
- 34-D raw keypoints (17 keypoints x 2 coordinates)
- 12-D engineered features:
  - Head direction (nose - ear center)
  - Body direction (shoulder - hip)
  - Left/right elbow angles
  - Ankle distance
  - Left/right knee angles
  - Shoulder width, hip width, ratio
- 3-D status (actual, interpolated, missing)

### Enriched Bbox Features (14-D)
- 4-D box displacements from first frame
- 2-D center displacements from first frame
- 2-D size normalized by first frame
- 1-D aspect ratio
- 1-D area normalized by first frame
- 2-D velocity (frame-to-frame center displacement)
- 2-D size change (frame-to-frame)

### Segmentation Crops
- Dual-frame (first + last observation frame)
- 224x224 pixel crops from Mask2Former semantic segmentation
- 19 Cityscapes classes

### Traffic Context (5-D)
- road_type, ped_crossing, ped_sign, stop_sign, traffic_light

### Vehicle Actions (5-D)
- One-hot encoded vehicle actions

## Training Details

- **Loss**: BCEWithLogitsLoss with pos_weight = raw_pos_weight x 0.25
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.0)
- **Scheduler**: Linear warmup (5 epochs) + Cosine annealing
- **SWA**: Average last 10 epochs
- **Batch size**: 32
- **Epochs**: 20
- **Gradient clipping**: max_norm=1.0

## Knowledge Distillation

### Output-Level KD
KL divergence between teacher and student soft predictions:

```
L_KD = alpha * T^2 * KL(p_T || p_S) + (1-alpha) * BCE(y, p_S)
```

### Representation-Level KD
MSE loss between teacher and student pose tokens:

```
L_repr = ||z_T - z_S||^2
```

### Combined KD

```
L = alpha_out * L_KD + alpha_repr * L_repr + (1-alpha_out-alpha_repr) * BCE
```

## Results

| Model | Params | AUC | F1 | Acc | P | R |
|-------|--------|-----|----|----|---|---|
| Teacher (B+S+P+T+V) | ~430K | 0.934 | 0.727 | 0.894 | 0.723 | 0.731 |
| Student (Conv1D, 64) | ~53K | 0.864 | 0.579 | 0.843 | 0.594 | 0.565 |
| Student (Conv1D+BBox, 64) | ~65K | 0.895 | 0.642 | 0.861 | 0.639 | 0.645 |

## File Structure

```
pose_to_intent/
├── extract_poses.py       # ViTPose keypoint extraction
├── extract_seg_maps.py    # Mask2Former segmentation extraction
├── models.py              # Teacher and student model definitions
├── data.py                # Data pipeline (JAAD loading, feature engineering)
├── train.py               # Training loop
├── evaluate.py            # Evaluation metrics
├── train_teacher.py       # Script to train teacher model
├── train_student.py       # Script to train student with/without KD
├── evaluate_model.py      # Script to evaluate a trained model
└── README.md              # This file
```

## Citation

