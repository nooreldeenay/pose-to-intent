"""Pose-to-Intent Evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5, criterion: nn.Module = None) -> Dict[str, float]:
    """Evaluate model on a data loader.

    Returns dict with: acc, auc, f1, precision, recall, tp, fp, fn, tn, loss (if criterion provided)
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            xb = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                  if k not in ('label', 'beh')}
            yb = batch['label'].to(device, non_blocking=True)

            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.sigmoid(logits).cpu().numpy()

            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item()
                n_batches += 1

            all_probs.append(probs)
            all_labels.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metrics = compute_metrics(probs, labels, threshold)
    if criterion is not None:
        metrics['loss'] = total_loss / max(n_batches, 1)

    return metrics


def get_predictions(model: nn.Module, loader: DataLoader,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Get per-sample probabilities and labels from a model + loader."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            xb = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                  if k not in ('label', 'beh')}
            yb = batch['label'].to(device, non_blocking=True)

            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(yb.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute classification metrics from probabilities and labels."""
    preds = (probs >= threshold).astype(int)

    acc = float((preds == labels).mean())

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = float(roc_auc_score(labels, probs))
        ap = float(average_precision_score(labels, probs))
    except ImportError:
        auc = 0.5
        ap = 0.5

    return {
        'acc': acc,
        'auc': auc,
        'ap': ap,
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }
