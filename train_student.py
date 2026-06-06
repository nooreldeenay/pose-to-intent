"""Train Student Model with Knowledge Distillation.

Student: Conv1D/GRU/Transformer (~53K params, AUC=0.864)
Teacher: TrafficVehicleTransformer (~430K params, AUC=0.934)

Usage:
    export JAAD_PATH=/path/to/JAAD
    python train_student.py                    # Train without KD
    python train_student.py --use_kd           # Train with KD
    python train_student.py --use_repr_kd      # Train with representation KD
"""

import os
import sys
import json
import random
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import build_data, compute_pose_features, get_jaad_path
from models import TeacherModel, StudentConv1D, StudentGRU, StudentTransformer
from evaluate import evaluate

warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

OUTPUT_DIR = os.path.expanduser(os.environ.get('SAVE_DIR', '~/ped_data/pose_to_intent/student'))
EPOCHS = 20
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_state_dicts(state_dicts):
    avg = {}
    for k in state_dicts[0]:
        avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
    return avg


# ===================================================================
# Flip Augmentation
# ===================================================================

_FLIP_KP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def flip_pose(pose):
    T, D = pose.shape
    out = pose.copy()
    if D >= 34:
        raw = out[:, :34].reshape(T, 17, 2)
        raw[:, :, 0] = -raw[:, :, 0]
        raw = raw[:, _FLIP_KP_MAP, :]
        out[:, :34] = raw.reshape(T, 34)
    if D >= 49:
        out[:, 34] = -out[:, 34]
        out[:, 36] = -out[:, 36]
        out[:, [38, 39]] = out[:, [39, 38]]
        out[:, [41, 42]] = out[:, [42, 41]]
    return out


def flip_bbox_enriched(bbox):
    out = bbox.copy()
    out[:, 0] = -out[:, 0]
    out[:, 2] = -out[:, 2]
    out[:, 4] = -out[:, 4]
    out[:, 10] = -out[:, 10]
    return out


# ===================================================================
# Dataset
# ===================================================================

class StudentDataset(torch.utils.data.Dataset):
    def __init__(self, data, use_bbox=False, augment=True):
        self.labels = data['crossing'].astype(np.float32)
        self.pose = data['pose'].astype(np.float32)
        self.use_bbox = use_bbox
        self.augment = augment

        self.teacher_pose = data['pose'].astype(np.float32)
        self.bbox = data['bbox_enriched'].astype(np.float32)
        self.seg = data['seg_crop_dual']
        self.traffic = data['traffic_context'].astype(np.float32)
        self.vehicle = data['vehicle_onehot'].astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        pose = self.pose[idx].copy()
        teacher_pose = self.teacher_pose[idx].copy()

        if self.augment and random.random() < 0.5:
            pose = flip_pose(pose)
            teacher_pose = flip_pose(teacher_pose)

        out = {
            'label': label,
            'pose': torch.from_numpy(pose),
            'teacher_pose': torch.from_numpy(teacher_pose),
            'bbox_enriched': torch.from_numpy(self.bbox[idx].copy()),
            'seg_crop_dual': torch.from_numpy(self.seg[idx].copy()).long(),
            'traffic_context': torch.from_numpy(self.traffic[idx].copy()),
            'vehicle_onehot': torch.from_numpy(self.vehicle[idx].copy()),
        }

        if self.use_bbox:
            bbox = self.bbox[idx].copy()
            if self.augment and random.random() < 0.5:
                bbox = flip_bbox_enriched(bbox)
            out['bbox_enriched'] = torch.from_numpy(bbox)

        return out


# ===================================================================
# Knowledge Distillation Loss
# ===================================================================

def kd_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.3):
    """Knowledge distillation loss for binary classification."""
    p_teacher = torch.sigmoid(teacher_logits / temperature)
    p_student = torch.sigmoid(student_logits / temperature)

    eps = 1e-6
    kd = (p_teacher * torch.log(p_teacher / (p_student + eps) + eps) +
          (1 - p_teacher) * torch.log((1 - p_teacher) / (1 - p_student + eps) + eps))
    kd = kd.mean() * (temperature ** 2)

    hard = F.binary_cross_entropy_with_logits(student_logits, labels)

    return alpha * kd + (1 - alpha) * hard


def kd_loss_with_repr(student_logits, teacher_logits, student_repr, teacher_repr,
                      labels, temperature=2.0, alpha_out=0.3, alpha_repr=0.3):
    """KD loss with both output-level and representation-level distillation."""
    p_teacher = torch.sigmoid(teacher_logits / temperature)
    p_student = torch.sigmoid(student_logits / temperature)
    eps = 1e-6
    kd_out = (p_teacher * torch.log(p_teacher / (p_student + eps) + eps) +
              (1 - p_teacher) * torch.log((1 - p_teacher) / (1 - p_student + eps) + eps))
    kd_out = kd_out.mean() * (temperature ** 2)

    kd_repr = F.mse_loss(student_repr, teacher_repr)

    hard = F.binary_cross_entropy_with_logits(student_logits, labels)

    return alpha_out * kd_out + alpha_repr * kd_repr + (1 - alpha_out - alpha_repr) * hard


# ===================================================================
# Training
# ===================================================================

def train_student(model, train_ds, val_ds, test_ds, save_dir, criterion,
                  teacher=None, temperature=3.0, alpha=0.5, use_repr_kd=False,
                  alpha_out=0.3, alpha_repr=0.3):
    set_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - 5)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    model = model.to(DEVICE)
    if teacher is not None:
        teacher = teacher.to(DEVICE)
        teacher.eval()

    best_auc = 0.0
    best_path = os.path.join(save_dir, 'best_model.pt')
    os.makedirs(save_dir, exist_ok=True)

    checkpoints = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            xb = {k: v.to(DEVICE) for k, v in batch.items() if k != 'label'}
            yb = batch['label'].to(DEVICE)

            student_logits = model(xb)

            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(xb)

                if use_repr_kd:
                    _, teacher_repr = teacher(xb, return_pose_token=True)
                    _, student_repr = model(xb, return_pose_token=True)
                    loss = kd_loss_with_repr(student_logits, teacher_logits,
                                             student_repr, teacher_repr, yb,
                                             temperature=temperature,
                                             alpha_out=alpha_out, alpha_repr=alpha_repr)
                else:
                    loss = kd_loss(student_logits, teacher_logits, yb,
                                   temperature=temperature, alpha=alpha)
            else:
                loss = criterion(student_logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, DEVICE)

        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpoints.append({
            'epoch': epoch,
            'state_dict': state_dict,
            'val_auc': val_metrics['auc'],
        })

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), best_path)

        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Ep {epoch:3d}/{EPOCHS} loss={train_loss:.4f} "
                  f"val_auc={val_metrics['auc']:.3f} f1={val_metrics['f1']:.3f} "
                  f"acc={val_metrics['acc']:.3f} "
                  f"p={val_metrics['precision']:.3f} r={val_metrics['recall']:.3f} "
                  f"lr={current_lr:.6f}")

    # SWA
    swa_dicts = [c['state_dict'] for c in checkpoints[-10:]]
    avg_dict = average_state_dicts(swa_dicts)
    model.load_state_dict(avg_dict)
    torch.save(model.state_dict(), os.path.join(save_dir, 'swa_model.pt'))

    test_metrics = evaluate(model, test_loader, DEVICE)
    return test_metrics


# ===================================================================
# Main
# ===================================================================

STUDENT_MODELS = {
    'conv1d': StudentConv1D,
    'gru': StudentGRU,
    'transformer': StudentTransformer,
}


def main():
    parser = argparse.ArgumentParser(description='Train student model')
    parser.add_argument('--model', type=str, default='conv1d',
                        choices=['conv1d', 'gru', 'transformer'],
                        help='Student model architecture')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--use_bbox', action='store_true',
                        help='Use bbox features alongside pose')
    parser.add_argument('--use_kd', action='store_true',
                        help='Use knowledge distillation')
    parser.add_argument('--use_repr_kd', action='store_true',
                        help='Use representation-level KD')
    parser.add_argument('--teacher_path', type=str, default=None,
                        help='Path to teacher checkpoint')
    parser.add_argument('--alpha_out', type=float, default=0.3,
                        help='Weight for output KD loss')
    parser.add_argument('--alpha_repr', type=float, default=0.3,
                        help='Weight for representation KD loss')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='KD temperature')
    parser.add_argument('--activation', type=str, default='none',
                        choices=['none', 'relu', 'gelu', 'leaky_relu'],
                        help='Activation function')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  STUDENT TRAINING — {args.model.upper()}")
    print("=" * 60)

    jaad_path = get_jaad_path()

    # Load data
    print("\nLoading data...")
    train_data = build_data(jaad_path, 'train', sample_type='all',
                            use_enriched_bbox=True, use_seg=True, seg_dual_frame=True,
                            use_pose=True, use_vehicle=True, use_traffic=True,
                            norm_method='bbox_size')
    val_data = build_data(jaad_path, 'val', sample_type='all',
                          use_enriched_bbox=True, use_seg=True, seg_dual_frame=True,
                          use_pose=True, use_vehicle=True, use_traffic=True,
                          norm_method='bbox_size')
    test_data = build_data(jaad_path, 'test', sample_type='all',
                           use_enriched_bbox=True, use_seg=True, seg_dual_frame=True,
                           use_pose=True, use_vehicle=True, use_traffic=True,
                           norm_method='bbox_size')

    # Compute 49-D pose features
    print("\nComputing pose features...")
    train_data['pose'] = compute_pose_features(train_data['pose'], train_data['pose_status'])
    val_data['pose'] = compute_pose_features(val_data['pose'], val_data['pose_status'])
    test_data['pose'] = compute_pose_features(test_data['pose'], test_data['pose_status'])

    # Load teacher if using KD
    teacher = None
    if args.use_kd or args.use_repr_kd:
        print("\nLoading teacher model...")
        teacher = TeacherModel()
        teacher_path = args.teacher_path or os.path.expanduser(
            '~/ped_data/pose_to_intent/teacher/seed_43/best_model.pt')
        if os.path.exists(teacher_path):
            teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
            teacher.eval()
            print(f"  Loaded teacher from {teacher_path}")
        else:
            print(f"  ERROR: Teacher not found at {teacher_path}")
            return

    # Create datasets
    train_ds = StudentDataset(train_data, use_bbox=args.use_bbox)
    val_ds = StudentDataset(val_data, use_bbox=False)
    test_ds = StudentDataset(test_data, use_bbox=False)

    # Build student
    model_cls = STUDENT_MODELS[args.model]
    model = model_cls(input_dim=49, hidden=args.hidden, use_bbox=args.use_bbox,
                      activation=args.activation)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Loss
    pos_count = int(train_data['crossing'].sum())
    neg_count = len(train_data['crossing']) - pos_count
    pos_weight = (neg_count / pos_count) * 0.25
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE))

    # Train
    save_dir = os.path.join(OUTPUT_DIR, f'{args.model}_h{args.hidden}' +
                            ('_bbox' if args.use_bbox else '') +
                            ('_kd' if args.use_kd else '') +
                            ('_reprkd' if args.use_repr_kd else '') +
                            f'_act{args.activation}_seed{args.seed}')

    metrics = train_student(model, train_ds, val_ds, test_ds, save_dir, criterion,
                            teacher=teacher, temperature=args.temperature,
                            use_repr_kd=args.use_repr_kd,
                            alpha_out=args.alpha_out, alpha_repr=args.alpha_repr)

    print(f"\n  Test: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
          f"Acc={metrics['acc']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")

    # Save results
    results = {
        'model': args.model,
        'hidden': args.hidden,
        'use_bbox': args.use_bbox,
        'use_kd': args.use_kd,
        'use_repr_kd': args.use_repr_kd,
        'activation': args.activation,
        'n_params': n_params,
        'metrics': metrics,
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
