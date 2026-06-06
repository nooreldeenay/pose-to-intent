"""Pose-to-Intent Training Loop."""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import JAADDataset
from evaluate import evaluate, get_predictions


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        xb = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ('label', 'beh')}
        yb = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(model: nn.Module,
          train_data: dict,
          val_data: Optional[dict],
          config: dict,
          save_dir: str,
          test_data: Optional[dict] = None) -> dict:
    """Full training loop.

    Args:
        model: PyTorch model
        train_data: dict from build_data()
        val_data: dict from build_data() or None
        config: training config
        save_dir: directory to save model

    Returns: dict with training history
    """
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 40)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    grad_clip = config.get('grad_clip', 1.0)
    warmup_epochs = config.get('warmup_epochs', 5)
    pos_weight_scale = config.get('pos_weight_scale', 1.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = JAADDataset(train_data, use_pose=True, use_seg=True,
                           use_enriched_bbox=True, use_seg_dual=True,
                           use_vehicle=True, use_traffic=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    val_loader = None
    if val_data is not None:
        val_ds = JAADDataset(val_data, use_pose=True, use_seg=True,
                             use_enriched_bbox=True, use_seg_dual=True,
                             use_vehicle=True, use_traffic=True)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

    test_loader = None
    if test_data is not None:
        test_ds = JAADDataset(test_data, use_pose=True, use_seg=True,
                              use_enriched_bbox=True, use_seg_dual=True,
                              use_vehicle=True, use_traffic=True)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

    pos_count = int(train_data['crossing'].sum())
    neg_count = len(train_data['crossing']) - pos_count
    raw_pos_weight = neg_count / pos_count
    pos_weight = raw_pos_weight * pos_weight_scale
    print(f"  Class weights: pos_weight={pos_weight:.2f} (raw={raw_pos_weight:.2f}, scale={pos_weight_scale})")

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"  Loss: BCE (pos_weight={pos_weight:.2f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if warmup_epochs > 0 and warmup_epochs < epochs:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = None

    model = model.to(device)

    print(f"\n  Training: {len(train_ds)} samples, {len(train_loader)} batches/epoch")
    print(f"  Device: {device}, Model: {model.__class__.__name__}")
    print(f"  batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, grad_clip={grad_clip}")

    best_val_auc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_metrics': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        history['train_loss'].append(train_loss)

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = lr

        elapsed = time.time() - t0

        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, criterion=criterion)
            history['val_metrics'].append(val_metrics)

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} | "
                      f"val: auc={val_metrics['auc']:.3f} f1={val_metrics['f1']:.3f} "
                      f"p={val_metrics['precision']:.3f} r={val_metrics['recall']:.3f} | "
                      f"lr={current_lr:.6f} | {elapsed:.1f}s")
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | "
                      f"lr={current_lr:.6f} | {elapsed:.1f}s")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pt'))

    if val_loader is not None and best_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=device, weights_only=True))
        print(f"\n  Loaded best model from epoch {best_epoch} (val_auc={best_val_auc:.4f})")

    if val_loader is not None:
        val_probs, val_labels = get_predictions(model, val_loader, device)
        np.save(os.path.join(save_dir, 'val_probs.npy'), val_probs)
        np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)
        print(f"  Saved val predictions: {len(val_labels)} samples")
    if test_loader is not None:
        test_probs, test_labels = get_predictions(model, test_loader, device)
        np.save(os.path.join(save_dir, 'test_probs.npy'), test_probs)
        np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)
        print(f"  Saved test predictions: {len(test_labels)} samples")

    history['best_epoch'] = best_epoch
    history['best_val_auc'] = best_val_auc
    return history
