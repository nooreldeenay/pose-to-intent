"""Evaluate a trained model on the test set.

Usage:
    python evaluate_model.py --model_path <path_to_checkpoint> --model_type <teacher|student_conv1d|student_gru|student_transformer>
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import build_data, compute_pose_features, JAADDataset
from models import build_model
from evaluate import evaluate, get_predictions

JAAD_PATH = '/home/nour/thesis_project/JAAD'


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['teacher', 'student_conv1d', 'student_gru', 'student_transformer'],
                        help='Model architecture')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden dimension (for student models)')
    parser.add_argument('--use_bbox', action='store_true',
                        help='Use bbox features (for student models)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n=== Loading test data ===")
    test_data = build_data(JAAD_PATH, 'test', sample_type='all',
                           use_enriched_bbox=True, use_seg=True, seg_dual_frame=True,
                           use_pose=True, use_vehicle=True, use_traffic=True,
                           norm_method='bbox_size')

    # Compute pose features
    print("\n=== Computing pose features ===")
    test_data['pose'] = compute_pose_features(test_data['pose'], test_data['pose_status'])

    # Build model
    print(f"\n=== Building {args.model_type} model ===")
    model = build_model(args.model_type, hidden=args.hidden, use_bbox=args.use_bbox)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Load checkpoint
    print(f"\n=== Loading checkpoint ===")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"  Loaded from {args.model_path}")

    # Create dataset
    test_ds = JAADDataset(test_data, use_pose=True, use_seg=True,
                          use_enriched_bbox=True, use_seg_dual=True,
                          use_vehicle=True, use_traffic=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Evaluate
    print("\n=== Evaluating ===")
    metrics = evaluate(model, test_loader, device)

    print(f"\n  Test Results:")
    print(f"    AUC:       {metrics['auc']:.4f}")
    print(f"    PR-AUC:    {metrics['ap']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    Accuracy:  {metrics['acc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")

    # Save predictions
    output_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)

    probs, labels = get_predictions(model, test_loader, device)
    np.save(os.path.join(output_dir, 'test_probs.npy'), probs)
    np.save(os.path.join(output_dir, 'test_labels.npy'), labels)

    results = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'n_params': n_params,
        'metrics': metrics,
    }
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {output_dir}")


if __name__ == '__main__':
    main()
