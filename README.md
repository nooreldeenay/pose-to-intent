# Pose to Intent

Pedestrian crossing prediction using pose, bbox, segmentation, traffic, and vehicle features on the JAAD dataset.

## Overview

This repository contains:

1. **Teacher Model** (TrafficVehicleTransformer): B+S+P+T+V with Transformer temporal processing (~430K params, AUC=0.934, F1=0.727)
2. **Student Models**: Lightweight models (~53K params) trained with/without knowledge distillation
   - StudentConv1D: Conv1D k=3 + k=7 (best, AUC=0.864)
   - StudentGRU: 2-layer GRU
   - StudentTransformer: 2-layer Transformer

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- OpenCV
- scikit-learn
- tqdm (optional)

## Data Setup

1. Clone the JAAD dataset to `~/thesis_project/JAAD/`
2. Extract pose data to `~/ped_data/jaad/poses/pose_set01.pkl`
3. Extract segmentation crops to `~/ped_data/jaad/seg_last_frame_mask2former/`

## Usage

### Train Teacher Model

```bash
python train_teacher.py
```

Trains the TrafficVehicleTransformer teacher model with SWA (average last 10 epochs).

### Train Student Model

```bash
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

### Evaluate Model

```bash
python evaluate_model.py --model_path <path_to_checkpoint> --model_type teacher
python evaluate_model.py --model_path <path_to_checkpoint> --model_type student_conv1d
```

## Model Architecture

### Teacher (TrafficVehicleTransformer)

```
Bbox Stream:    Linear(14→64) → Transformer(2 layers, 4 heads) → mean pooling
Seg Stream:     Embedding(19→16) → CNN(3 layers) → Attention pooling → Linear(64→64)
Pose Stream:    Linear(49→64) → Conv1D(k=3) + Conv1D(k=7) → Linear(128→64)
Traffic Stream: Linear(5→64) → Transformer(2 layers, 4 heads) → mean pooling
Vehicle Stream: Linear(5→64) → Transformer(2 layers, 4 heads) → mean pooling
Classifier:     Linear(64×5→1)
```

### Student (StudentConv1D)

```
Pose Stream:    Linear(49→64) → Conv1D(k=3) + Conv1D(k=7) → Linear(128→64)
Classifier:     Linear(64→1)
```

## Feature Engineering

### Pose Features (49-D)
- 34-D raw keypoints (17 keypoints × 2 coordinates)
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
- 224×224 pixel crops from Mask2Former semantic segmentation
- 19 Cityscapes classes

### Traffic Context (5-D)
- road_type, ped_crossing, ped_sign, stop_sign, traffic_light

### Vehicle Actions (5-D)
- One-hot encoded vehicle actions

## Training Details

- **Loss**: BCEWithLogitsLoss with pos_weight = raw_pos_weight × 0.25
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.0)
- **Scheduler**: Linear warmup (5 epochs) + Cosine annealing
- **SWA**: Average last 10 epochs
- **Batch size**: 32
- **Epochs**: 20
- **Gradient clipping**: max_norm=1.0

## Knowledge Distillation

### Output-Level KD
KL divergence between teacher and student soft predictions:
$$\mathcal{L}_{KD} = \alpha \cdot T^2 \cdot KL(p_T \| p_S) + (1-\alpha) \cdot BCE(y, p_S)$$

### Representation-Level KD
MSE loss between teacher and student pose tokens:
$$\mathcal{L}_{repr} = ||z_T - z_S||^2$$

### Combined KD
$$\mathcal{L} = \alpha_{out} \cdot \mathcal{L}_{KD} + \alpha_{repr} \cdot \mathcal{L}_{repr} + (1-\alpha_{out}-\alpha_{repr}) \cdot BCE$$

## Results

| Model | Params | AUC | F1 | Acc | P | R |
|-------|--------|-----|----|----|---|---|
| Teacher (B+S+P+T+V) | ~430K | 0.934 | 0.727 | 0.894 | 0.723 | 0.731 |
| Student (Conv1D, 64) | ~53K | 0.864 | 0.579 | 0.843 | 0.594 | 0.565 |
| Student (Conv1D+BBox, 64) | ~65K | 0.895 | 0.642 | 0.861 | 0.639 | 0.645 |

## File Structure

```
pose_to_intent/
├── models.py           # Teacher and student model definitions
├── data.py             # Data pipeline (JAAD loading, feature engineering)
├── train.py            # Training loop
├── evaluate.py         # Evaluation metrics
├── train_teacher.py    # Script to train teacher model
├── train_student.py    # Script to train student with/without KD
├── evaluate_model.py   # Script to evaluate a trained model
└── README.md           # This file
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{pedestrian_action_benchmark,
  title={Pedestrian Action Recognition Benchmark},
  author={...},
  year={2024}
}
```
