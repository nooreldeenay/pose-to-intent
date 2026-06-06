"""Pose-to-Intent Models.

Teacher: TrafficVehicleTransformer (B+S+P+T+V, ~430K params, AUC=0.934)
Student: Conv1D/GRU/Transformer (~53K params, AUC=0.864)
"""

import torch
import torch.nn as nn


# ===================================================================
# Temporal Processing Modules
# ===================================================================

class GRUProcessor(nn.Module):
    """GRU for temporal processing."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.gru = nn.GRU(output_dim, output_dim, batch_first=True)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.norm(self.proj(x))
        _, h = self.gru(x)
        return self.norm2(h.squeeze(0))


class TransformerProcessor(nn.Module):
    """Transformer for temporal processing."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, nhead=4, dim_feedforward=output_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.norm(self.proj(x))
        x = self.transformer(x)
        return self.norm2(x.mean(dim=1))


# ===================================================================
# Teacher Model: TrafficVehicleTransformer (B+S+P+T+V)
# ===================================================================

class TeacherModel(nn.Module):
    """B+S+P+T+V with Transformer temporal processing.

    Input streams:
        - bbox_enriched: (N, T, 14) enriched bbox features
        - seg_crop_dual: (N, 2, 224, 224) dual-frame segmentation crops
        - pose: (N, T, 49) engineered pose features (raw34 + 12 engineered + 3 status)
        - traffic_context: (N, T, 5) traffic context features
        - vehicle_onehot: (N, T, 5) vehicle action one-hot

    Output: (N,) crossing logits
    """

    def __init__(self, hidden=64, num_classes=19, embed_dim=16):
        super().__init__()
        self.hidden = hidden

        # Bbox stream: Transformer
        self.pos_embed = nn.Parameter(torch.randn(1, 16, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.bbox_proj = nn.Linear(14, hidden)
        self.bbox_norm = nn.LayerNorm(hidden)

        # Seg stream: CNN with attention pooling
        self.seg_embed = nn.Embedding(num_classes, embed_dim)
        self.seg_norm = nn.LayerNorm(embed_dim)
        self.seg_cnn = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )
        self.seg_attn = nn.Sequential(nn.Linear(hidden, 128),
                                      nn.Tanh(), nn.Linear(128, 1))
        self.seg_proj = nn.Linear(hidden, hidden)
        self.seg_norm2 = nn.LayerNorm(hidden)

        # Pose stream: Conv1D k=3 + k=7 (no_relu_all)
        self.pose_proj = nn.Linear(49, 64)
        self.pose_norm = nn.LayerNorm(64)
        self.pose_drop = nn.Dropout(0.1)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
        )
        self.conv_fuse = nn.Sequential(
            nn.Linear(64 * 2, hidden),
            nn.LayerNorm(hidden),
        )

        # Traffic + Vehicle streams: Transformer
        self.traffic_proc = TransformerProcessor(5, hidden)
        self.vehicle_proc = TransformerProcessor(5, hidden)

        # Classifier head
        self.head = nn.Linear(hidden * 5, 1)

    def forward(self, x, return_pose_token=False):
        # Bbox stream
        bx = self.bbox_norm(self.bbox_proj(x['bbox_enriched'])) + self.pos_embed
        bbox_tok = self.transformer(bx).mean(dim=1)

        # Seg stream
        seg = self.seg_norm(self.seg_embed(x['seg_crop_dual'][:, 1])).permute(0, 3, 1, 2)
        feat = self.seg_cnn(seg)
        N, C, H, W = feat.shape
        feat_t = feat.permute(0, 2, 3, 1).reshape(N, H * W, C)
        w = torch.softmax(self.seg_attn(feat_t), dim=1)
        seg_tok = self.seg_norm2(self.seg_proj((feat_t * w).sum(dim=1)))

        # Pose stream
        pose = self.pose_drop(self.pose_norm(self.pose_proj(x['pose'])))
        pose_t = pose.permute(0, 2, 1)
        c3 = self.conv3(pose_t).mean(dim=2)
        c7 = self.conv7(pose_t).mean(dim=2)
        pose_tok = self.conv_fuse(torch.cat([c3, c7], -1))

        # Traffic + Vehicle streams
        traffic_tok = self.traffic_proc(x['traffic_context'])
        vehicle_tok = self.vehicle_proc(x['vehicle_onehot'])

        # Fuse all streams
        fused = torch.cat([bbox_tok, seg_tok, pose_tok, traffic_tok, vehicle_tok], -1)
        logit = self.head(fused).squeeze(-1)

        if return_pose_token:
            return logit, pose_tok
        return logit


# ===================================================================
# Student Models
# ===================================================================

class StudentConv1D(nn.Module):
    """Conv1D student (default, matches teacher's pose stream).

    Input: pose (N, T, 49) engineered pose features
    Output: (N,) crossing logits
    """
    def __init__(self, input_dim=49, hidden=64, use_bbox=False, activation='none'):
        super().__init__()
        self.use_bbox = use_bbox

        self.pose_proj = nn.Linear(input_dim, hidden)
        self.pose_norm = nn.LayerNorm(hidden)
        self.pose_drop = nn.Dropout(0.1)

        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
        )

        if activation == 'relu':
            self.conv3.add_module('relu', nn.ReLU(inplace=True))
            self.conv7.add_module('relu', nn.ReLU(inplace=True))
        elif activation == 'gelu':
            self.conv3.add_module('gelu', nn.GELU())
            self.conv7.add_module('gelu', nn.GELU())
        elif activation == 'leaky_relu':
            self.conv3.add_module('leaky', nn.LeakyReLU(inplace=True))
            self.conv7.add_module('leaky', nn.LeakyReLU(inplace=True))

        self.conv_fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )

        if use_bbox:
            self.bbox_proj = nn.Linear(14, hidden)
            self.bbox_norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden * 2, 1)
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, x, return_pose_token=False):
        pose = self.pose_drop(self.pose_norm(self.pose_proj(x['pose'])))
        pose_t = pose.permute(0, 2, 1)
        c3 = self.conv3(pose_t).mean(dim=2)
        c7 = self.conv7(pose_t).mean(dim=2)
        pose_tok = self.conv_fuse(torch.cat([c3, c7], -1))

        if self.use_bbox:
            bbox_tok = self.bbox_norm(self.bbox_proj(x['bbox_enriched']))
            bbox_tok = bbox_tok[:, -1]
            logit = self.head(torch.cat([pose_tok, bbox_tok], -1)).squeeze(-1)
        else:
            logit = self.head(pose_tok).squeeze(-1)

        if return_pose_token:
            return logit, pose_tok
        return logit


class StudentGRU(nn.Module):
    """GRU student.

    Input: pose (N, T, 49) engineered pose features
    Output: (N,) crossing logits
    """
    def __init__(self, input_dim=49, hidden=64, use_bbox=False, activation='none'):
        super().__init__()
        self.use_bbox = use_bbox

        self.pose_proj = nn.Linear(input_dim, hidden)
        self.pose_norm = nn.LayerNorm(hidden)
        self.pose_drop = nn.Dropout(0.1)

        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          dropout=0.1)

        if use_bbox:
            self.bbox_proj = nn.Linear(14, hidden)
            self.bbox_norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden * 2, 1)
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, x, return_pose_token=False):
        pose = self.pose_drop(self.pose_norm(self.pose_proj(x['pose'])))
        _, h = self.gru(pose)
        pose_tok = h[-1]

        if self.use_bbox:
            bbox_tok = self.bbox_norm(self.bbox_proj(x['bbox_enriched']))
            bbox_tok = bbox_tok[:, -1]
            logit = self.head(torch.cat([pose_tok, bbox_tok], -1)).squeeze(-1)
        else:
            logit = self.head(pose_tok).squeeze(-1)

        if return_pose_token:
            return logit, pose_tok
        return logit


class StudentTransformer(nn.Module):
    """Transformer student.

    Input: pose (N, T, 49) engineered pose features
    Output: (N,) crossing logits
    """
    def __init__(self, input_dim=49, hidden=64, use_bbox=False, activation='none'):
        super().__init__()
        self.use_bbox = use_bbox

        self.pose_proj = nn.Linear(input_dim, hidden)
        self.pose_norm = nn.LayerNorm(hidden)
        self.pose_drop = nn.Dropout(0.1)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, hidden) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        if use_bbox:
            self.bbox_proj = nn.Linear(14, hidden)
            self.bbox_norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden * 2, 1)
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, x, return_pose_token=False):
        pose = self.pose_norm(self.pose_proj(x['pose'])) + self.pos_embed
        pose_tok = self.transformer(pose).mean(dim=1)

        if self.use_bbox:
            bbox_tok = self.bbox_norm(self.bbox_proj(x['bbox_enriched']))
            bbox_tok = bbox_tok[:, -1]
            logit = self.head(torch.cat([pose_tok, bbox_tok], -1)).squeeze(-1)
        else:
            logit = self.head(pose_tok).squeeze(-1)

        if return_pose_token:
            return logit, pose_tok
        return logit


# ===================================================================
# Model Registry
# ===================================================================

MODEL_REGISTRY = {
    'teacher': TeacherModel,
    'student_conv1d': StudentConv1D,
    'student_gru': StudentGRU,
    'student_transformer': StudentTransformer,
}


def build_model(model_name, **kwargs):
    """Build a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)
