"""Pose-to-Intent Data Pipeline.

Self-contained data pipeline for JAAD crossing prediction.
Loads raw JAAD data, windows tracks, computes features, and creates PyTorch datasets.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 1920, 1080
IMG_PATH = os.path.expanduser('~/thesis_project/JAAD/images')

OBS_LENGTH = 16
TIME_TO_EVENT = [30, 60]
OVERLAP = 0.8

POSE_PATH = os.path.expanduser('~/ped_data/jaad/poses/pose_set01.pkl')
SEG_CACHE_PATH = os.path.expanduser('~/ped_data/jaad/seg_last_frame_mask2former')

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


# ---------------------------------------------------------------------------
# Step 1.1: Load raw data
# ---------------------------------------------------------------------------

def load_raw_data(jaad_path: str, split: str, min_track_size: int = 76,
                  sample_type: str = 'all') -> dict:
    """Load raw JAAD data for a split."""
    sys.path.insert(0, jaad_path)
    from jaad_data import JAAD

    print(f"  Loading JAAD database...", end=' ', flush=True)
    imdb = JAAD(data_path=jaad_path)
    print("done")

    data_opts = {
        'fstride': 1,
        'sample_type': sample_type,
        'subset': 'default',
        'data_split_type': 'default',
        'seq_type': 'crossing',
        'min_track_size': min_track_size,
    }

    print(f"  Generating {split} sequences...", end=' ', flush=True)
    raw = imdb.generate_data_trajectory_sequence(split, **data_opts)
    n_tracks = len(raw['bbox'])
    print(f"done ({n_tracks} tracks)")

    assert n_tracks == len(raw['image']), f"Mismatch: {n_tracks} bboxes vs {len(raw['image'])} images"
    assert n_tracks == len(raw['pid']), f"Mismatch: {n_tracks} bboxes vs {len(raw['pid'])} pids"
    assert n_tracks == len(raw['activities']), f"Mismatch: {n_tracks} bboxes vs {len(raw['activities'])} activities"

    return raw


# ---------------------------------------------------------------------------
# Step 1.2: Window into observation sequences
# ---------------------------------------------------------------------------

def window_tracks(raw: dict, obs_length: int = OBS_LENGTH,
                  time_to_event: list = TIME_TO_EVENT,
                  overlap: float = OVERLAP,
                  future_steps: int = 0) -> dict:
    """Window tracks into fixed-length observation sequences."""
    bbox_seqs = raw['bbox']
    center_seqs = raw['center']
    pid_seqs = raw['pid']
    activity_seqs = raw['activities']
    image_seqs = raw['image']
    vehicle_seqs = raw.get('vehicle_act', None)
    traffic_seqs = raw.get('traffic', None)

    olap_res = int((1 - overlap) * obs_length)
    olap_res = max(1, olap_res)

    all_box_raw = []
    all_center_raw = []
    all_crossing = []
    all_pid = []
    all_image = []
    all_tte = []
    all_vehicle = []
    all_traffic = []
    all_box_future = []

    for i in range(len(bbox_seqs)):
        seq_len = len(bbox_seqs[i])
        start_idx = seq_len - obs_length - time_to_event[1]
        end_idx = seq_len - obs_length - time_to_event[0]

        if start_idx < 0:
            continue

        for w in range(start_idx, end_idx + 1, olap_res):
            box_window = bbox_seqs[i][w:w + obs_length]
            center_window = center_seqs[i][w:w + obs_length]
            image_window = image_seqs[i][w:w + obs_length]
            pid_window = pid_seqs[i][w:w + obs_length]

            crossing = activity_seqs[i][w + obs_length - 1][0]
            tte = seq_len - (w + obs_length)

            all_box_raw.append(box_window)
            all_center_raw.append(center_window)
            all_crossing.append(crossing)
            all_pid.append(pid_window[0][0] if hasattr(pid_window[0], '__iter__') else pid_window[0])
            all_image.append(image_window)
            all_tte.append(tte)

            if vehicle_seqs is not None:
                vehicle_window = vehicle_seqs[i][w:w + obs_length]
                all_vehicle.append(vehicle_window)

            if traffic_seqs is not None:
                traffic_window = traffic_seqs[i][w:w + obs_length]
                all_traffic.append(traffic_window)

            if future_steps > 0:
                future_start = w + obs_length
                future_end = future_start + future_steps
                if future_end <= seq_len:
                    box_future = bbox_seqs[i][future_start:future_end]
                else:
                    box_future = bbox_seqs[i][future_start:]
                    while len(box_future) < future_steps:
                        box_future.append(box_future[-1])
                all_box_future.append(box_future)

    print(f"  Converting {len(all_crossing)} windows to arrays...", end=' ', flush=True)
    box_raw = np.array(all_box_raw, dtype=np.float32)
    center_raw = np.array(all_center_raw, dtype=np.float32)
    crossing = np.array(all_crossing, dtype=np.int64)
    ped_id = np.array(all_pid)
    beh = np.array(['b' in str(p) for p in all_pid], dtype=np.int64)
    image = np.array(all_image)
    tte = np.array(all_tte, dtype=np.int64)
    print("done")

    box_norm = box_raw - box_raw[:, 0:1, :]
    center_norm = center_raw - center_raw[:, 0:1, :]

    N = len(crossing)
    assert box_raw.shape == (N, obs_length, 4)
    assert box_norm.shape == (N, obs_length, 4)
    assert center_raw.shape == (N, obs_length, 2)
    assert center_norm.shape == (N, obs_length, 2)
    assert crossing.shape == (N,)
    assert image.shape == (N, obs_length)

    result = {
        'box_raw': box_raw,
        'box': box_norm,
        'center_raw': center_raw,
        'center': center_norm,
        'crossing': crossing,
        'beh': beh,
        'ped_id': ped_id,
        'image': image,
        'tte': tte,
    }

    if all_vehicle:
        vehicle_act = np.array(all_vehicle, dtype=np.int64)
        result['vehicle_act'] = vehicle_act

    if all_traffic:
        N = len(all_traffic)
        T = len(all_traffic[0])
        traffic_arr = np.zeros((N, T, 5), dtype=np.float32)
        for i in range(N):
            for t in range(T):
                frame_data = all_traffic[i][t]
                if isinstance(frame_data, list):
                    frame_data = frame_data[0] if frame_data else {}
                traffic_arr[i, t, 0] = frame_data.get('road_type', 0)
                traffic_arr[i, t, 1] = frame_data.get('ped_crossing', 0)
                traffic_arr[i, t, 2] = frame_data.get('ped_sign', 0)
                traffic_arr[i, t, 3] = frame_data.get('stop_sign', 0)
                traffic_arr[i, t, 4] = frame_data.get('traffic_light', 0)
        result['traffic'] = traffic_arr

    if all_box_future:
        box_raw_future = np.array(all_box_future, dtype=np.float32)
        result['box_raw_future'] = box_raw_future

    return result


# ---------------------------------------------------------------------------
# Enriched bbox features
# ---------------------------------------------------------------------------

NORM_METHODS = ['none', 'image', 'bbox', 'bbox_size', 'sample_minmax', 'minmax', 'zscore']


def compute_enriched_bbox_features(data: dict, norm_method: str = 'none') -> np.ndarray:
    """Compute enriched bbox features from raw pixel bboxes.

    Features (14-D per timestep):
      1. box_norm: (x1, y1, x2, y2) displacements from first frame (4-D)
      2. center_norm: (cx, cy) displacements from first frame (2-D)
      3. size_norm: (w, h) normalized by first frame's size (2-D)
      4. aspect_ratio: w/h (1-D)
      5. area_norm: w*h normalized by first frame's area (1-D)
      6. velocity: (dcx, dcy) frame-to-frame center displacement (2-D)
      7. size_change: (dw, dh) frame-to-frame size change (2-D)
    """
    assert norm_method in NORM_METHODS, f"Unknown norm_method: {norm_method}"

    print(f"  Computing enriched bbox features (norm={norm_method})...", end=' ', flush=True)
    box_raw = data['box_raw']
    N, T, _ = box_raw.shape

    x1 = box_raw[:, :, 0]
    y1 = box_raw[:, :, 1]
    x2 = box_raw[:, :, 2]
    y2 = box_raw[:, :, 3]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    w = np.clip(x2 - x1, 1, None)
    h = np.clip(y2 - y1, 1, None)

    cx0 = cx[:, 0:1]
    cy0 = cy[:, 0:1]
    w0 = w[:, 0:1]
    h0 = h[:, 0:1]

    box_norm = box_raw - box_raw[:, 0:1, :]
    center_norm = np.stack([cx - cx0, cy - cy0], axis=-1)

    if norm_method == 'none':
        pass
    elif norm_method == 'image':
        box_norm[:, :, 0] /= IMG_W
        box_norm[:, :, 1] /= IMG_H
        box_norm[:, :, 2] /= IMG_W
        box_norm[:, :, 3] /= IMG_H
        center_norm[:, :, 0] /= IMG_W
        center_norm[:, :, 1] /= IMG_H
    elif norm_method == 'bbox':
        box_norm[:, :, 0] /= w0
        box_norm[:, :, 1] /= h0
        box_norm[:, :, 2] /= w0
        box_norm[:, :, 3] /= h0
        center_norm[:, :, 0] /= w0
        center_norm[:, :, 1] /= h0
    elif norm_method == 'bbox_size':
        box_norm[:, :, 0] /= w0
        box_norm[:, :, 1] /= h0
        box_norm[:, :, 2] /= w0
        box_norm[:, :, 3] /= h0
        center_norm[:, :, 0] /= w0
        center_norm[:, :, 1] /= h0
    elif norm_method == 'minmax':
        for feat in [box_norm, center_norm]:
            fmin = feat.min()
            fmax = feat.max()
            if fmax > fmin:
                feat[:] = (feat - fmin) / (fmax - fmin)
    elif norm_method == 'zscore':
        for feat in [box_norm, center_norm]:
            fmean = feat.mean()
            fstd = feat.std()
            if fstd > 0:
                feat[:] = (feat - fmean) / fstd

    size_norm = np.stack([w / w0, h / h0], axis=-1)
    aspect_ratio = (w / h)[..., np.newaxis]
    area0 = w0 * h0
    area_norm = (w * h / area0)[..., np.newaxis]

    velocity = np.zeros((N, T, 2), dtype=np.float32)
    velocity[:, 1:, 0] = cx[:, 1:] - cx[:, :-1]
    velocity[:, 1:, 1] = cy[:, 1:] - cy[:, :-1]
    if norm_method == 'bbox_size':
        velocity[:, :, 0] /= w0[:, 0:1]
        velocity[:, :, 1] /= h0[:, 0:1]
    else:
        velocity[:, :, 0] /= IMG_W
        velocity[:, :, 1] /= IMG_H

    size_change = np.zeros((N, T, 2), dtype=np.float32)
    size_change[:, 1:, 0] = w[:, 1:] - w[:, :-1]
    size_change[:, 1:, 1] = h[:, 1:] - h[:, :-1]
    size_change[:, :, 0] /= w0
    size_change[:, :, 1] /= h0

    features = np.concatenate([
        box_norm, center_norm, size_norm, aspect_ratio,
        area_norm, velocity, size_change,
    ], axis=-1)

    if norm_method == 'sample_minmax':
        for i in range(N):
            fmin = features[i].min()
            fmax = features[i].max()
            if fmax > fmin:
                features[i] = (features[i] - fmin) / (fmax - fmin)

    assert features.shape == (N, T, 14)
    assert not np.isnan(features).any(), "NaN in features"
    assert not np.isinf(features).any(), "Inf in features"

    print(f"  done ({N} samples, {T} timesteps, 14 features)")
    return features


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def interpolate_poses(poses: np.ndarray):
    """Fill all-zero pose frames by linear interpolation."""
    out = poses.copy()
    N, T, D = out.shape
    status = np.zeros((N, T, 3), dtype=np.float32)

    for i in range(N):
        valid_mask = np.abs(out[i]).sum(axis=-1) > 0
        valid_indices = np.where(valid_mask)[0]

        status[i, valid_mask, 0] = 1.0

        if len(valid_indices) == 0:
            status[i, :, 2] = 1.0
            continue

        first_valid = valid_indices[0]
        if first_valid > 0:
            out[i, :first_valid] = out[i, first_valid]
            status[i, :first_valid, 1] = 1.0

        last_valid = valid_indices[-1]
        if last_valid < T - 1:
            out[i, last_valid + 1:] = out[i, last_valid]
            status[i, last_valid + 1:, 1] = 1.0

        for j in range(len(valid_indices) - 1):
            t_start = valid_indices[j]
            t_end = valid_indices[j + 1]
            gap = t_end - t_start - 1
            if gap <= 0:
                continue
            for step in range(1, gap + 1):
                alpha = step / (gap + 1)
                out[i, t_start + step] = (1 - alpha) * out[i, t_start] + alpha * out[i, t_end]
                status[i, t_start + step, 1] = 1.0

    return out, status


def load_poses(image_seqs: np.ndarray, ped_ids: np.ndarray,
               pose_path: str = POSE_PATH,
               interp: bool = True):
    """Load and optionally interpolate pose data."""
    print(f"  Loading poses from {pose_path}...", end=' ', flush=True)
    with open(pose_path, 'rb') as f:
        try:
            pose_data = pickle.load(f)
        except Exception:
            pose_data = pickle.load(f, encoding='bytes')
    print("done")

    set_id = 'set01'
    if set_id in pose_data and isinstance(pose_data[set_id], dict):
        pose_data = pose_data[set_id]

    N, T = image_seqs.shape
    poses = np.zeros((N, T, 34), dtype=np.float32)

    print(f"  Extracting poses for {N} samples...", end=' ', flush=True)
    for i in range(N):
        vid_id = image_seqs[i][0].split('/')[-2]
        for t in range(T):
            frame_name = image_seqs[i][t].split('/')[-1].split('.')[0]
            pid = str(ped_ids[i])
            key = f"{frame_name}_{pid}"

            if vid_id in pose_data and key in pose_data[vid_id]:
                poses[i, t] = np.array(pose_data[vid_id][key], dtype=np.float32)
    print("done")

    if interp:
        print(f"  Interpolating missing poses...", end=' ', flush=True)
        poses, status = interpolate_poses(poses)
        n_actual = int(status[:, :, 0].sum())
        n_interp = int(status[:, :, 1].sum())
        n_missing = int(status[:, :, 2].sum())
        n_total = N * T
        print(f"done (actual={n_actual/n_total:.1%}, interp={n_interp/n_total:.1%}, missing={n_missing/n_total:.1%})")
    else:
        status = np.zeros((N, T, 3), dtype=np.float32)
        valid = np.abs(poses).sum(axis=-1) > 0
        status[valid, 0] = 1.0
        status[~valid, 2] = 1.0

    return poses, status


# ---------------------------------------------------------------------------
# Pose feature engineering (49-D)
# ---------------------------------------------------------------------------

COCO_KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

_FLIP_KP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _get_kp(poses, name):
    idx = COCO_KP[name] * 2
    return poses[..., idx], poses[..., idx + 1]


def _dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_pose_features(poses, pose_status):
    """Compute 49-D pose features: 34 raw + 12 engineered + 3 status."""
    N, T, _ = poses.shape
    raw = poses.copy()
    engineered = np.zeros((N, T, 12), dtype=np.float32)

    for i in range(N):
        for t in range(T):
            if pose_status[i, t, 2] == 1.0:
                continue

            nx, ny = _get_kp(poses[i, t], 'nose')
            lex, ley = _get_kp(poses[i, t], 'left_ear')
            rex, rey = _get_kp(poses[i, t], 'right_ear')
            mid_ex = (lex + rex) / 2
            mid_ey = (ley + rey) / 2
            engineered[i, t, 0] = nx - mid_ex
            engineered[i, t, 1] = ny - mid_ey

            lsx, lsy = _get_kp(poses[i, t], 'left_shoulder')
            rsx, rsy = _get_kp(poses[i, t], 'right_shoulder')
            lhx, lhy = _get_kp(poses[i, t], 'left_hip')
            rhx, rhy = _get_kp(poses[i, t], 'right_hip')
            scx = (lsx + rsx) / 2
            scy = (lsy + rsy) / 2
            hcx = (lhx + rhx) / 2
            hcy = (lhy + rhy) / 2
            engineered[i, t, 2] = scx - hcx
            engineered[i, t, 3] = scy - hcy

            lwx, lwy = _get_kp(poses[i, t], 'left_wrist')
            lex2, ley2 = _get_kp(poses[i, t], 'left_elbow')
            v1x, v1y = lsx - lex2, lsy - ley2
            v2x, v2y = lwx - lex2, lwy - ley2
            dot = v1x * v2x + v1y * v2y
            n1 = np.sqrt(v1x**2 + v1y**2) + 1e-6
            n2 = np.sqrt(v2x**2 + v2y**2) + 1e-6
            engineered[i, t, 4] = np.clip(dot / (n1 * n2), -1, 1)

            rwx, rwy = _get_kp(poses[i, t], 'right_wrist')
            rex2, rey2 = _get_kp(poses[i, t], 'right_elbow')
            v1x, v1y = rsx - rex2, rsy - rey2
            v2x, v2y = rwx - rex2, rwy - rey2
            dot = v1x * v2x + v1y * v2y
            n1 = np.sqrt(v1x**2 + v1y**2) + 1e-6
            n2 = np.sqrt(v2x**2 + v2y**2) + 1e-6
            engineered[i, t, 5] = np.clip(dot / (n1 * n2), -1, 1)

            lax, lay = _get_kp(poses[i, t], 'left_ankle')
            rax, ray = _get_kp(poses[i, t], 'right_ankle')
            engineered[i, t, 6] = _dist(lax, lay, rax, ray)

            lkx, lky = _get_kp(poses[i, t], 'left_knee')
            v1x, v1y = lhx - lkx, lhy - lky
            v2x, v2y = lax - lkx, lay - lky
            dot = v1x * v2x + v1y * v2y
            n1 = np.sqrt(v1x**2 + v1y**2) + 1e-6
            n2 = np.sqrt(v2x**2 + v2y**2) + 1e-6
            engineered[i, t, 7] = np.clip(dot / (n1 * n2), -1, 1)

            rkx, rky = _get_kp(poses[i, t], 'right_knee')
            v1x, v1y = rhx - rkx, rhy - rky
            v2x, v2y = rax - rkx, ray - rky
            dot = v1x * v2x + v1y * v2y
            n1 = np.sqrt(v1x**2 + v1y**2) + 1e-6
            n2 = np.sqrt(v2x**2 + v2y**2) + 1e-6
            engineered[i, t, 8] = np.clip(dot / (n1 * n2), -1, 1)

            engineered[i, t, 9] = _dist(lsx, lsy, rsx, rsy)
            engineered[i, t, 10] = _dist(lhx, lhy, rhx, rhy)
            sw = engineered[i, t, 9]
            hw = engineered[i, t, 10]
            engineered[i, t, 11] = sw / (hw + 1e-6)

    return np.concatenate([raw, engineered, pose_status], axis=-1)


# ---------------------------------------------------------------------------
# Seg crop loading
# ---------------------------------------------------------------------------

def load_seg_crops(image_seqs: np.ndarray, box_raw: np.ndarray,
                   seg_cache_path: str = SEG_CACHE_PATH,
                   dual_frame: bool = False,
                   frame_indices: list = None,
                   debug_vis: bool = False) -> np.ndarray:
    """Load segmentation crops from cache."""
    N, T, _ = image_seqs.shape

    if dual_frame:
        indices = [0, T - 1]
    elif frame_indices is not None:
        indices = frame_indices
    else:
        indices = [T - 1]

    n_frames = len(indices)
    crops = np.zeros((N, n_frames, 224, 224), dtype=np.uint8)

    for i in range(N):
        vid_id = image_seqs[i][0].split('/')[-2]
        for fi, t in enumerate(indices):
            frame_name = image_seqs[i][t].split('/')[-1].split('.')[0]
            cache_path = os.path.join(seg_cache_path, vid_id, f'{frame_name}.npy')

            if os.path.exists(cache_path):
                seg_map = np.load(cache_path)
            else:
                seg_map = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

            x1, y1, x2, y2 = box_raw[i, t].astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(IMG_W, x2)
            y2 = min(IMG_H, y2)

            if x2 > x1 and y2 > y1:
                crop = seg_map[y1:y2, x1:x2]
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_NEAREST)
            else:
                crop = np.zeros((224, 224), dtype=np.uint8)

            crops[i, fi] = crop

    print(f"  Loaded seg crops: {crops.shape} (dual_frame={dual_frame})")
    return crops


# ---------------------------------------------------------------------------
# Vehicle & traffic loading
# ---------------------------------------------------------------------------

def load_vehicle_actions(data: dict) -> np.ndarray:
    """Load vehicle actions as one-hot encoding."""
    if 'vehicle_act' not in data:
        N = len(data['crossing'])
        T = OBS_LENGTH
        return np.zeros((N, T, 5), dtype=np.float32)

    vehicle_act = data['vehicle_act']
    N, T, _ = vehicle_act.shape

    onehot = np.zeros((N, T, 5), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            act = int(vehicle_act[i, t, 0]) if vehicle_act[i, t].ndim > 0 else int(vehicle_act[i, t])
            act = min(max(act, 0), 4)
            onehot[i, t, act] = 1.0

    print(f"  Loaded vehicle actions: {onehot.shape}")
    return onehot


def load_traffic_context(data: dict) -> np.ndarray:
    """Load traffic context features."""
    if 'traffic' not in data:
        N = len(data['crossing'])
        T = OBS_LENGTH
        return np.zeros((N, T, 5), dtype=np.float32)

    traffic = data['traffic']
    print(f"  Loaded traffic context: {traffic.shape}")
    return traffic


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JAADDataset(Dataset):
    """PyTorch-compatible dataset for JAAD crossing prediction."""

    def __init__(self, data: dict, use_pose: bool = False, use_seg: bool = False,
                 use_enriched_bbox: bool = False, use_seg_dual: bool = False,
                 use_vehicle: bool = False, use_traffic: bool = False):
        self.crossing = data['crossing']
        self.beh = data.get('beh', np.zeros(len(self.crossing), dtype=np.int64))
        self.use_pose = use_pose
        self.use_seg = use_seg
        self.use_seg_dual = use_seg_dual
        self.use_enriched_bbox = use_enriched_bbox
        self.use_vehicle = use_vehicle
        self.use_traffic = use_traffic

        if use_enriched_bbox:
            assert 'bbox_enriched' in data
            self.bbox_enriched = data['bbox_enriched']
        else:
            self.box = data['box']
            self.center = data['center']

        if use_pose:
            assert 'pose' in data
            self.pose = data['pose']

        if use_seg_dual:
            assert 'seg_crop_dual' in data
            self.seg_crop_dual = data['seg_crop_dual']
        elif use_seg:
            assert 'seg_crop' in data
            self.seg_crop = data['seg_crop']

        if use_vehicle:
            assert 'vehicle_onehot' in data
            self.vehicle_onehot = data['vehicle_onehot']

        if use_traffic:
            assert 'traffic_context' in data
            self.traffic_context = data['traffic_context']

    def __len__(self):
        return len(self.crossing)

    def __getitem__(self, idx):
        out = {
            'label': torch.tensor(self.crossing[idx], dtype=torch.float32),
            'beh': torch.tensor(self.beh[idx], dtype=torch.float32),
        }

        if self.use_enriched_bbox:
            out['bbox_enriched'] = torch.tensor(self.bbox_enriched[idx], dtype=torch.float32)
        else:
            out['box'] = torch.tensor(self.box[idx], dtype=torch.float32)
            out['center'] = torch.tensor(self.center[idx], dtype=torch.float32)

        if self.use_pose:
            out['pose'] = torch.tensor(self.pose[idx], dtype=torch.float32)

        if self.use_seg_dual:
            out['seg_crop_dual'] = torch.tensor(self.seg_crop_dual[idx], dtype=torch.int64)
        elif self.use_seg:
            out['seg_crop'] = torch.tensor(self.seg_crop[idx], dtype=torch.int64)

        if self.use_vehicle:
            out['vehicle_onehot'] = torch.tensor(self.vehicle_onehot[idx], dtype=torch.float32)

        if self.use_traffic:
            out['traffic_context'] = torch.tensor(self.traffic_context[idx], dtype=torch.float32)

        return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_data(jaad_path: str, split: str, sample_type: str = 'all',
               use_pose: bool = False, use_seg: bool = False,
               use_enriched_bbox: bool = False,
               use_vehicle: bool = False,
               use_traffic: bool = False,
               seg_dual_frame: bool = False,
               norm_method: str = 'none') -> dict:
    """Full pipeline: load raw -> window -> add modalities."""
    print(f"\n  === Building {split} data ===")

    min_track_size = OBS_LENGTH + TIME_TO_EVENT[1]
    raw = load_raw_data(jaad_path, split, min_track_size=min_track_size,
                        sample_type=sample_type)

    data = window_tracks(raw, obs_length=OBS_LENGTH,
                         time_to_event=TIME_TO_EVENT,
                         overlap=OVERLAP)

    if use_enriched_bbox:
        data['bbox_enriched'] = compute_enriched_bbox_features(data, norm_method=norm_method)

    if use_pose:
        poses, pose_status = load_poses(data['image'], data['ped_id'],
                                        pose_path=POSE_PATH, interp=True)
        data['pose'] = poses
        data['pose_status'] = pose_status

    if use_seg or seg_dual_frame:
        if seg_dual_frame:
            data['seg_crop_dual'] = load_seg_crops(data['image'], data['box_raw'],
                                                   seg_cache_path=SEG_CACHE_PATH,
                                                   dual_frame=True)
        else:
            data['seg_crop'] = load_seg_crops(data['image'], data['box_raw'],
                                              seg_cache_path=SEG_CACHE_PATH)

    if use_vehicle:
        data['vehicle_onehot'] = load_vehicle_actions(data)

    if use_traffic:
        data['traffic_context'] = load_traffic_context(data)

    print(f"  === {split} data ready: {len(data['crossing'])} samples ===\n")
    return data
