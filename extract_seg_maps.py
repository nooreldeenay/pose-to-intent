"""Extract and cache Mask2Former segmentation maps for the full JAAD dataset.

Processes all frames referenced in the JAAD dataset, caches results as uint8 numpy arrays.
Skips already-computed frames unless --regen_data is set.

Usage:
    export JAAD_PATH=/path/to/JAAD
    export SEG_CACHE_DIR=~/ped_data/jaad/seg_maps
    python extract_seg_maps.py
    python extract_seg_maps.py --regen_data
"""

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor


IMG_W, IMG_H = 1920, 1080

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

CITYSCAPES_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def get_env_or_exit(name):
    """Get environment variable or exit with error."""
    val = os.environ.get(name)
    if val is None:
        print(f"Error: Set {name} environment variable first.")
        print(f"  export {name}=/path/to/{name.lower()}")
        sys.exit(1)
    return val


def load_seg_model(device):
    """Load Mask2Former model for semantic segmentation."""
    model_name = 'facebook/mask2former-swin-large-cityscapes-semantic'
    print(f"Loading Mask2Former ({model_name}) on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor


def extract_seg_map(model, processor, img_path, device):
    """Run Mask2Former on a single frame, return uint8 seg map."""
    frame = cv2.imread(img_path)
    if frame is None:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(frame.shape[0], frame.shape[1])]
    )[0]
    return seg.cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Extract Mask2Former seg maps for JAAD')
    parser.add_argument('--regen_data', action='store_true',
                        help='Recompute all seg maps (ignore cache)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Data splits to process')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for extraction')
    args = parser.parse_args()

    jaad_path = get_env_or_exit('JAAD_PATH')
    seg_cache_dir = os.path.expanduser(os.environ.get('SEG_CACHE_DIR', '~/ped_data/jaad/seg_maps'))

    sys.path.insert(0, jaad_path)
    from jaad_data import JAAD

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = load_seg_model(device)

    imdb = JAAD(data_path=jaad_path)

    data_opts = {
        'fstride': 1,
        'sample_type': 'all',
        'subset': 'default',
        'data_split_type': 'default',
        'seq_type': 'crossing',
        'min_track_size': 15,
    }

    total_frames = 0
    cached_frames = 0
    processed_frames = 0

    for split in args.splits:
        print(f"\nProcessing split: {split}")
        raw = imdb.generate_data_trajectory_sequence(split, **data_opts)

        num_samples = len(raw['image'])
        for idx in tqdm(range(num_samples), desc=f"{split} tracks"):
            seq_images = raw['image'][idx]
            T = len(seq_images)

            for t in range(T):
                img_path = seq_images[t]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(jaad_path, img_path)

                # Build cache path
                parts = img_path.split('/')
                vid_id = parts[-2]
                frame_name = parts[-1].split('.')[0]
                cache_path = os.path.join(seg_cache_dir, vid_id, f'{frame_name}.npy')

                total_frames += 1

                # Skip if already cached
                if os.path.exists(cache_path) and not args.regen_data:
                    cached_frames += 1
                    continue

                # Extract seg map
                seg_map = extract_seg_map(model, processor, img_path, device)

                # Save to cache
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, seg_map, allow_pickle=False)
                processed_frames += 1

    print(f"\nDone.")
    print(f"Total frames: {total_frames}")
    print(f"Cached (skipped): {cached_frames}")
    print(f"Newly processed: {processed_frames}")
    print(f"Cache saved to: {seg_cache_dir}")


if __name__ == '__main__':
    main()
