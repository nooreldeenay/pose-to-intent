"""Extract and cache ViTPose keypoints for the full JAAD dataset.

Processes all splits (train/val/test), caches results to $POSE_CACHE_DIR/pose_set01.pkl.
Skips already-computed frames unless --regen_data is set.

Usage:
    export JAAD_PATH=/path/to/JAAD
    export POSE_CACHE_DIR=~/ped_data/jaad/poses
    python extract_poses.py
    python extract_poses.py --regen_data
"""

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, VitPoseForPoseEstimation


def get_env_or_exit(name):
    """Get environment variable or exit with error."""
    val = os.environ.get(name)
    if val is None:
        print(f"Error: Set {name} environment variable first.")
        print(f"  export {name}=/path/to/{name.lower()}")
        sys.exit(1)
    return val


def load_pose_cache(pose_cache_file):
    """Load existing pose cache or return empty dict."""
    if os.path.exists(pose_cache_file):
        with open(pose_cache_file, 'rb') as f:
            try:
                nested = pickle.load(f)
            except Exception:
                nested = pickle.load(f, encoding='bytes')
        flat = {}
        for set_id, vids in nested.items():
            for vid_id, frames in vids.items():
                for key, vec in frames.items():
                    flat[f"{set_id}/{vid_id}/{key}"] = vec
        return flat
    return {}


def save_pose_cache(flat_cache, pose_cache_file):
    """Save pose cache to disk, converting back to nested structure."""
    os.makedirs(os.path.dirname(pose_cache_file), exist_ok=True)
    nested = {}
    for full_key, vec in flat_cache.items():
        parts = full_key.split('/', 2)
        if len(parts) == 3:
            set_id, vid_id, key = parts
            if set_id not in nested:
                nested[set_id] = {}
            if vid_id not in nested[set_id]:
                nested[set_id][vid_id] = {}
            nested[set_id][vid_id][key] = vec
    with open(pose_cache_file, 'wb') as f:
        pickle.dump(nested, f)


def extract_pose(model, processor, frame, bbox, device):
    """Extract 34-D pose vector (17 keypoints x 2) normalized to [0, 1]."""
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])

    coco_box = np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float32)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, boxes=[coco_box], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pose_results = processor.post_process_pose_estimation(
            outputs, boxes=[coco_box], threshold=0.3
        )

    if pose_results and len(pose_results[0]) > 0:
        person_pose = pose_results[0][0]
        kpts = person_pose["keypoints"].cpu().numpy()
        scores = person_pose["scores"].cpu().numpy()

        if len(kpts) == 17 and np.any(scores > 0.3):
            kpts[:, 0] /= img_w
            kpts[:, 1] /= img_h
            return kpts.flatten().tolist()

    return [0.0] * 34


def main():
    parser = argparse.ArgumentParser(description='Extract ViTPose keypoints for JAAD')
    parser.add_argument('--model', type=str, default='usyd-community/vitpose-base-simple',
                        help='ViTPose model name')
    parser.add_argument('--regen_data', action='store_true',
                        help='Recompute all poses (ignore cache)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Data splits to process')
    args = parser.parse_args()

    jaad_path = get_env_or_exit('JAAD_PATH')
    pose_cache_dir = os.path.expanduser(os.environ.get('POSE_CACHE_DIR', '~/ped_data/jaad/poses'))
    pose_cache_file = os.path.join(pose_cache_dir, 'pose_set01.pkl')

    sys.path.insert(0, jaad_path)
    from jaad_data import JAAD

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading ViTPose ({args.model}) on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = VitPoseForPoseEstimation.from_pretrained(args.model)
    model.to(device)
    model.eval()

    if not args.regen_data:
        cache = load_pose_cache(pose_cache_file)
        if cache:
            print(f"Loaded {len(cache)} cached entries.")
    else:
        cache = {}
        print("Starting fresh (regen_data=True).")

    set_id = 'set01'
    if set_id not in cache:
        cache[set_id] = {}

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
            seq_bboxes = raw['bbox'][idx]
            seq_pids = raw['pid'][idx]
            T = len(seq_images)

            for t in range(T):
                img_path = seq_images[t]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(jaad_path, img_path)

                parts = img_path.split('/')
                vid_id = parts[-2]
                frame_name = parts[-1].split('.')[0]
                pid = seq_pids[t][0]
                cache_key = f"{set_id}/{vid_id}/{frame_name}_{pid}"

                total_frames += 1

                if cache_key in cache and not args.regen_data:
                    cached_frames += 1
                    continue

                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                bbox = seq_bboxes[t]
                pose_vec = extract_pose(model, processor, frame, bbox, device)
                cache[cache_key] = pose_vec
                processed_frames += 1

                if processed_frames % 500 == 0:
                    save_pose_cache(cache, pose_cache_file)

    save_pose_cache(cache, pose_cache_file)

    print(f"\nDone.")
    print(f"Total frames: {total_frames}")
    print(f"Cached (skipped): {cached_frames}")
    print(f"Newly processed: {processed_frames}")
    print(f"Cache saved to: {pose_cache_file}")


if __name__ == '__main__':
    main()
