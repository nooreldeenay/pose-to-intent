"""Train Teacher Model (TrafficVehicleTransformer).

Teacher: B+S+P+T+V with Transformer temporal processing.
~430K params, AUC=0.934, F1=0.727 (3-seed, JAAD all).

Usage:
    export JAAD_PATH=/path/to/JAAD
    python train_teacher.py
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from data import build_data, compute_pose_features, get_jaad_path
from models import TeacherModel
from train import train_one_epoch
from evaluate import evaluate, get_predictions

warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

SAVE_ROOT = os.path.expanduser(os.environ.get('SAVE_DIR', '~/ped_data/pose_to_intent/teacher'))
SEEDS = [43]
EPOCHS = 20
BATCH_SIZE = 32


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_state_dicts(state_dicts):
    """Average multiple state dicts."""
    avg = {}
    for k in state_dicts[0]:
        avg[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
    return avg


def train_model(model, train_ds, val_ds, test_ds, save_dir, criterion, device, seed):
    set_seed(seed)

    from torch.utils.data import DataLoader
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

    model = model.to(device)
    best_auc = 0.0
    val_best_path = os.path.join(save_dir, 'val_best_model.pt')
    swa_best_path = os.path.join(save_dir, 'best_model.pt')
    os.makedirs(save_dir, exist_ok=True)

    checkpoints = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpoints.append({
            'epoch': epoch,
            'state_dict': state_dict,
            'val_auc': val_metrics['auc'],
        })

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), val_best_path)

        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Ep {epoch:3d}/{EPOCHS} loss={train_loss:.4f} "
                  f"val_auc={val_metrics['auc']:.3f} f1={val_metrics['f1']:.3f} "
                  f"acc={val_metrics['acc']:.3f} "
                  f"p={val_metrics['precision']:.3f} r={val_metrics['recall']:.3f} "
                  f"lr={current_lr:.6f}")

    # SWA: average last 10 epochs
    swa_dicts = [c['state_dict'] for c in checkpoints[-10:]]
    swa_avg = average_state_dicts(swa_dicts)
    model.load_state_dict(swa_avg)
    model = model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), swa_best_path)

    # Save predictions
    val_probs, val_labels = get_predictions(model, val_loader, device)
    test_probs, test_labels = get_predictions(model, test_loader, device)
    np.save(os.path.join(save_dir, 'val_probs.npy'), val_probs)
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(save_dir, 'test_probs.npy'), test_probs)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)

    return model, test_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    jaad_path = get_jaad_path()

    print("\n=== Loading data ===")
    train_data = build_data(jaad_path, 'train', use_enriched_bbox=True,
                            seg_dual_frame=True, norm_method='bbox_size',
                            use_pose=True, use_traffic=True, use_vehicle=True)
    val_data = build_data(jaad_path, 'val', use_enriched_bbox=True,
                          seg_dual_frame=True, norm_method='bbox_size',
                          use_pose=True, use_traffic=True, use_vehicle=True)
    test_data = build_data(jaad_path, 'test', use_enriched_bbox=True,
                           seg_dual_frame=True, norm_method='bbox_size',
                           use_pose=True, use_traffic=True, use_vehicle=True)

    # Compute 49-D pose features
    print("\n=== Engineering pose features ===")
    train_data['pose'] = compute_pose_features(train_data['pose'], train_data['pose_status'])
    val_data['pose'] = compute_pose_features(val_data['pose'], val_data['pose_status'])
    test_data['pose'] = compute_pose_features(test_data['pose'], test_data['pose_status'])

    pos_count = int(train_data['crossing'].sum())
    neg_count = len(train_data['crossing']) - pos_count
    raw_pos_weight = neg_count / pos_count
    pos_weight = raw_pos_weight * 0.25
    print(f"  pos_weight={pos_weight:.2f} (raw={raw_pos_weight:.2f})")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))

    from data import JAADDataset
    train_ds = JAADDataset(train_data, use_pose=True, use_seg=True,
                           use_enriched_bbox=True, use_seg_dual=True,
                           use_vehicle=True, use_traffic=True)
    val_ds = JAADDataset(val_data, use_pose=True, use_seg=True,
                         use_enriched_bbox=True, use_seg_dual=True,
                         use_vehicle=True, use_traffic=True)
    test_ds = JAADDataset(test_data, use_pose=True, use_seg=True,
                          use_enriched_bbox=True, use_seg_dual=True,
                          use_vehicle=True, use_traffic=True)

    all_results = {}

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        set_seed(seed)
        model = TeacherModel()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        save_dir = os.path.join(SAVE_ROOT, f'seed_{seed}')
        t0 = time.time()
        _, metrics = train_model(model, train_ds, val_ds, test_ds,
                                 save_dir, criterion, device, seed)
        elapsed = time.time() - t0

        print(f"  Seed {seed}: AUC={metrics['auc']:.4f} PR-AUC={metrics['ap']:.4f} "
              f"F1={metrics['f1']:.4f} Acc={metrics['acc']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"({elapsed:.1f}s)")

        all_results[seed] = {
            'auc': metrics['auc'],
            'ap': metrics['ap'],
            'f1': metrics['f1'],
            'accuracy': metrics['acc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'time_s': elapsed,
        }

    # Summary
    aucs = [r['auc'] for r in all_results.values()]
    f1s = [r['f1'] for r in all_results.values()]
    print(f"\n  Summary: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}, "
          f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    # Save results
    import json
    os.makedirs(SAVE_ROOT, exist_ok=True)
    with open(os.path.join(SAVE_ROOT, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {SAVE_ROOT}")


if __name__ == '__main__':
    main()
