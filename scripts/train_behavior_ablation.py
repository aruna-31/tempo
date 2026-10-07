#!/usr/bin/env python3
"""
TEMPO — Behaviour Recognition Research Model Ablation & Generalization Pipeline
================================================================================
Implements a reproducible, rigorous research ablation and improvement study comparing:

Architecture Ablation Matrix:
  A. ResNet-18 + RNN
  B. ResNet-18 + GRU
  C. ResNet-18 + BiGRU
  D. ResNet-18 + BiGRU + Temporal Attention
  E. ResNet-50 + GRU
  F. ResNet-50 + BiGRU
  G. ResNet-50 + BiGRU + Temporal Attention

Controlled Generalization Study:
  1. Baseline augmentation vs Stronger classroom augmentation
     (ColorJitter, GaussianBlur, Additive Noise, RandomAffine, RandomResizedCrop, RandomErasing)
  2. Backbone Fine-Tuning Strategy:
     - Frozen backbone (pretrained ImageNet)
     - Last ResNet block fine-tuned with lower LR (1e-5 / 2e-5)
     - Temporal head with higher LR (1e-3) / Discriminative learning rates

Enforces:
  - Exact train/val/test splits (from index_repaired.csv & sequences_repaired.json)
  - Identical sequence construction (T=16, 2.0 sampling FPS)
  - Class-balanced loss across the 5 observable classes
  - Reproducible random seeds (seed=42)
  - Validation-only checkpoint selection (early stopping guided strictly by Val Macro-F1)
  - Evaluation on both Validation AND untouched frozen Test set
  - Full per-epoch loss/F1 curves, confusion matrices, and per-class metrics
  - Safe caching of extracted backbone features to avoid redundant forward passes
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "models" / "training_data" / "dataset"
INDEX_PATH = DATA_DIR / "index_repaired.csv"
SEQ_PATH = DATA_DIR / "sequences_repaired.json"

RESEARCH_DIR = BASE_DIR / "storage" / "research" / "behavior_ablation"
CACHE_DIR = RESEARCH_DIR / "cache"
CKPT_DIR = RESEARCH_DIR / "checkpoints"
METRICS_DIR = RESEARCH_DIR / "metrics"
CURVES_DIR = RESEARCH_DIR / "curves"
CONFIGS_DIR = RESEARCH_DIR / "configs"

PROD_WEIGHTS_PATH = BASE_DIR / "models" / "classroom_temporal_model.pth"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tempo.ml.ablation")

# ── 5 Observable Classes ──
CLASSES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]
NUM_CLASSES = len(CLASSES)
CL2I = {c: i for i, c in enumerate(CLASSES)}

# ── Normalization ──
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ── Transforms ──
baseline_train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.RandomHorizontalFlip(p=0.3),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

strong_train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
    T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.90, 1.10)),
    T.RandomHorizontalFlip(p=0.3),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    T.RandomErasing(p=0.35, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


# ── Seed Initialization ──
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── In-Memory Crop Dataset for Spatial Fine-Tuning ──
class InMemoryCropDataset(Dataset):
    def __init__(self, tensors: List[torch.Tensor], labels: List[int], augment: bool = True):
        self.tensors = tensors
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = self.tensors[idx]
        l = self.labels[idx]
        if self.augment and random.random() < 0.35:
            t = t + torch.randn_like(t) * 0.015
        return t, l


# ── Sequence Feature Dataset ──
class SequenceFeatureDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        target_seq_len: int = 16,
        augment: bool = False,
        strong_aug: bool = False,
    ):
        self.samples = samples
        self.target_seq_len = target_seq_len
        self.augment = augment
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feats = item["features"]  # (T_orig, feat_dim)
        t_orig = feats.shape[0]

        if t_orig == self.target_seq_len:
            seq = feats.copy()
        elif t_orig > self.target_seq_len:
            indices = np.linspace(0, t_orig - 1, self.target_seq_len, dtype=int)
            seq = feats[indices].copy()
        else:
            pad_count = self.target_seq_len - t_orig
            repeat_pad = np.repeat(feats[-1:], pad_count, axis=0)
            seq = np.concatenate([feats, repeat_pad], axis=0)

        if self.augment:
            noise_std = 0.025 if self.strong_aug else 0.015
            if random.random() < 0.40:
                seq = seq + np.random.normal(0, noise_std, seq.shape).astype(np.float32)

            # Temporal jitter / frame dropout
            if self.target_seq_len > 4 and random.random() < (0.40 if self.strong_aug else 0.25):
                j_idx = random.randint(1, self.target_seq_len - 1)
                seq[j_idx] = seq[j_idx - 1]

        x = torch.from_numpy(seq).float()
        y = torch.tensor(item["label"], dtype=torch.long)
        return x, y, item.get("is_back_row", False), item.get("video_id", "")


# ── Model Architectures ──
class TemporalAttention(nn.Module):
    """Additive Temporal Self-Attention over sequence timesteps."""

    def __init__(self, feature_dim: int = 512, attention_dim: int = 128):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn_net(x)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(x * weights, dim=1)  # (B, feature_dim)
        return context, weights.squeeze(-1)


class UnifiedAblationModel(nn.Module):
    """
    Unified Temporal Sequence Classifier supporting all ablation variants:
      Backbones: ResNet-18 (512-dim), ResNet-50 (2048-dim with projection to 512-dim)
      Temporal Modules: RNN, GRU, BiGRU, BiGRU + Temporal Attention
    """

    def __init__(
        self,
        backbone_type: str = "resnet18",
        temporal_type: str = "GRU",
        use_attention: bool = False,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.temporal_type = temporal_type.upper()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Spatial feature input dimension
        self.in_feature_dim = 2048 if self.backbone_type == "resnet50" else 512

        # ResNet-50 projection layer
        if self.backbone_type == "resnet50":
            self.projection = nn.Sequential(
                nn.Linear(2048, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            temporal_in_dim = 512
        else:
            self.projection = nn.Identity()
            temporal_in_dim = 512

        # Temporal sequence model
        is_bidirectional = "BI" in self.temporal_type
        base_rnn_type = self.temporal_type.replace("BI", "")

        if base_rnn_type == "RNN":
            self.temporal = nn.RNN(
                input_size=temporal_in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="relu",
                bidirectional=is_bidirectional,
            )
        else:  # GRU
            self.temporal = nn.GRU(
                input_size=temporal_in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=is_bidirectional,
            )

        self.temporal_out_dim = hidden_dim * 2 if is_bidirectional else hidden_dim

        # Temporal attention
        if self.use_attention:
            self.attention = TemporalAttention(
                feature_dim=self.temporal_out_dim, attention_dim=128
            )
        else:
            self.attention = None

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_out_dim, 128),
            nn.LayerNorm(128) if (is_bidirectional or self.backbone_type == "resnet50") else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, T, in_feature_dim)
        """
        # Feature projection (if ResNet-50)
        proj = self.projection(x)

        # Temporal sequence processing
        rnn_out, _ = self.temporal(proj)  # (B, T, temporal_out_dim)

        # Context aggregation
        if self.use_attention:
            context, attn_weights = self.attention(rnn_out)
        else:
            context = rnn_out[:, -1, :]  # last timestep
            attn_weights = None

        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits


# ── Metrics Calculation ──
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []
    supports = []

    for i in range(NUM_CLASSES):
        tp = int(cm[i][i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        sup = int(cm[i, :].sum())

        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

        precisions.append(pr)
        recalls.append(rc)
        f1s.append(f1)
        supports.append(sup)

        per_class[CLASSES[i]] = {
            "precision": round(pr, 4),
            "recall": round(rc, 4),
            "f1": round(f1, 4),
            "support": sup,
        }

    total_samples = max(1, len(y_true))
    acc = float((y_true == y_pred).sum()) / total_samples
    macro_f1 = float(np.mean(f1s))
    macro_pr = float(np.mean(precisions))
    macro_rc = float(np.mean(recalls))
    weighted_f1 = float(sum(f1 * sup for f1, sup in zip(f1s, supports)) / total_samples)

    pred_dist = {CLASSES[i]: int((y_pred == i).sum()) for i in range(NUM_CLASSES)}

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_pr, 4),
        "macro_recall": round(macro_rc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": pred_dist,
    }


# ── Spatial Backbone Fine-Tuning & Feature Extraction ──
def fine_tune_spatial_backbone(
    backbone_type: str,
    train_crops: List[Dict[str, Any]],
    device: torch.device,
    augmentation: str = "baseline",
    samples_per_class: int = 400,
    epochs: int = 3,
) -> nn.Module:
    """
    Fine-tunes layer4 and linear classification head on balanced training crops.
    Supports baseline and stronger classroom augmentations.
    """
    logger.info(
        f"Fine-tuning {backbone_type.upper()} backbone (layer4 + head) | "
        f"Augmentation: {augmentation} | Samples/Class: {samples_per_class}..."
    )

    by_class = defaultdict(list)
    for c in train_crops:
        lbl = c.get("filtered_tempo_label", c.get("tempo_label"))
        by_class[lbl].append(c)

    balanced_crops = []
    for cls_name, items in by_class.items():
        sample_k = min(len(items), samples_per_class)
        balanced_crops.extend(random.sample(items, sample_k))
    random.shuffle(balanced_crops)

    transform = strong_train_transform if augmentation == "strong" else baseline_train_transform

    tensors, labels = [], []
    for c in balanced_crops:
        full_p = DATA_DIR / c["relative_path"]
        if full_p.exists():
            try:
                img = Image.open(full_p).convert("RGB")
                tensors.append(transform(img))
                lbl = c.get("filtered_tempo_label", c.get("tempo_label"))
                labels.append(CL2I[lbl])
            except Exception:
                pass

    if backbone_type == "resnet50":
        try:
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            base_model = resnet50(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, NUM_CLASSES)
        lr_bb = 2e-5
        lr_head = 2e-4
    else:
        try:
            base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            base_model = resnet18(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, NUM_CLASSES)
        lr_bb = 5e-5
        lr_head = 1e-4

    base_model.to(device)

    # Freeze earlier layers, train layer4 + fc
    for name, p in base_model.named_parameters():
        if "layer4" in name or "fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    counts = Counter(labels)
    tot = len(labels)
    weights = torch.tensor([
        tot / (NUM_CLASSES * max(1, counts[i])) for i in range(NUM_CLASSES)
    ]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    layer4_params = [p for n, p in base_model.named_parameters() if "layer4" in n and p.requires_grad]
    fc_params = [p for n, p in base_model.named_parameters() if "fc" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": lr_bb},
        {"params": fc_params, "lr": lr_head},
    ], weight_decay=1e-4)

    loader = DataLoader(
        InMemoryCropDataset(tensors, labels, augment=True),
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )

    t0 = time.time()
    for ep in range(epochs):
        base_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = base_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += int((out.argmax(dim=1) == y).sum().item())
            total += len(y)

        logger.info(
            f"  Backbone Ep {ep+1}/{epochs}: loss={total_loss/max(1, total):.4f}, "
            f"acc={correct/max(1, total):.4f} ({time.time()-t0:.1f}s)"
        )

    logger.info(f"Spatial fine-tuning finished in {time.time()-t0:.1f}s.")
    return base_model


def get_or_extract_features(
    backbone_name: str,
    backbone_model: nn.Module,
    relative_paths: List[str],
    device: torch.device,
    force_extract: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extracts and caches spatial features in memory-efficient binary .npy format.
    Eliminates Python pickle serialization overhead and MemoryError.
    """
    cache_npy = CACHE_DIR / f"{backbone_name}_features.npy"
    cache_pt = CACHE_DIR / f"{backbone_name}_features.pt"
    paths_json = CACHE_DIR / f"{backbone_name}_paths.json"

    # 1. Check if .npy cache exists
    if cache_npy.exists() and not force_extract:
        logger.info(f"Loading cached features from {cache_npy}...")
        try:
            mat = np.load(cache_npy)
            if mat.shape[0] == len(relative_paths):
                logger.info(f"Loaded all {len(relative_paths)} features directly from {cache_npy.name} (shape {mat.shape}).")
                return {p: mat[i] for i, p in enumerate(relative_paths)}
        except Exception as e:
            logger.warning(f"Failed to load {cache_npy}: {e}, checking .pt fallback...")

    # 2. Check if valid .pt cache exists and convert to .npy
    if cache_pt.exists() and os.path.getsize(cache_pt) > 100000 and not force_extract:
        logger.info(f"Checking legacy .pt cache at {cache_pt}...")
        try:
            cached_dict = torch.load(cache_pt, map_location="cpu", weights_only=False)
            if all(p in cached_dict for p in relative_paths):
                feat_dim = cached_dict[relative_paths[0]].shape[0]
                mat = np.zeros((len(relative_paths), feat_dim), dtype=np.float32)
                for i, p in enumerate(relative_paths):
                    mat[i] = cached_dict[p]
                np.save(cache_npy, mat)
                logger.info(f"Converted {cache_pt.name} to {cache_npy.name} ({mat.shape}).")
                return {p: mat[i] for i, p in enumerate(relative_paths)}
        except Exception as e:
            logger.warning(f"Could not use legacy .pt cache: {e}")

    # 3. Extract fresh features directly into a pre-allocated numpy matrix
    feat_dim = 2048 if "resnet50" in backbone_name.lower() else 512
    if hasattr(backbone_model, "fc"):
        feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
    else:
        feature_extractor = backbone_model
    feature_extractor.eval()
    feature_extractor.to(device)

    features_mat = np.zeros((len(relative_paths), feat_dim), dtype=np.float32)
    path_to_idx = {p: i for i, p in enumerate(relative_paths)}

    bs = 128
    t0 = time.time()
    logger.info(f"Extracting features for {len(relative_paths):,} crops using {backbone_name} (dim={feat_dim})...")

    def _load_single(p_rel: str):
        full_p = DATA_DIR / p_rel
        if full_p.exists():
            try:
                img = Image.open(full_p).convert("RGB")
                return p_rel, eval_transform(img)
            except Exception:
                pass
        return p_rel, None

    for i in range(0, len(relative_paths), bs):
        batch_paths = relative_paths[i : i + bs]
        with ThreadPoolExecutor(max_workers=8) as ex:
            loaded_items = list(ex.map(_load_single, batch_paths))

        valid_paths = [p for p, t in loaded_items if t is not None]
        tensors = [t for p, t in loaded_items if t is not None]

        if not tensors:
            continue

        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch_t)
            if feats.dim() > 2:
                feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1).cpu().numpy()

        for j, p in enumerate(valid_paths):
            idx = path_to_idx[p]
            features_mat[idx] = feats[j].astype(np.float32)

        if (i + bs) % 2000 == 0 or (i + bs) >= len(relative_paths):
            logger.info(
                f"  Extracted {min(i+bs, len(relative_paths)):,} / {len(relative_paths):,} crops "
                f"({time.time()-t0:.1f}s, {min(i+bs, len(relative_paths))/max(0.1, time.time()-t0):.1f} crops/s)..."
            )

    logger.info(f"Feature extraction complete: {features_mat.shape}. Saving to {cache_npy}...")
    np.save(cache_npy, features_mat)
    with open(paths_json, "w", encoding="utf-8") as f:
        json.dump(relative_paths, f)

    return {p: features_mat[i] for i, p in enumerate(relative_paths)}


# ── Training Procedure for a Single Ablation Model ──
def train_single_ablation_model(
    exp_id: str,
    backbone_type: str,
    temporal_type: str,
    use_attention: bool,
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    device: torch.device,
    epochs: int = 30,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    strong_aug: bool = False,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes training and evaluation for one specific model configuration.
    Guidance rules:
      - Early stopping guided strictly by Validation Macro-F1.
      - Class-weighted CrossEntropyLoss.
      - Save best checkpoint and per-epoch curves.
      - Evaluate on untouched held-out test set at the end.
    """
    logger.info(f"\n{'='*75}\n  Training Experiment: {exp_id} ({backbone_type.upper()} + {temporal_type}{'+Attn' if use_attention else ''})\n{'='*75}")

    model = UnifiedAblationModel(
        backbone_type=backbone_type,
        temporal_type=temporal_type,
        use_attention=use_attention,
        hidden_dim=256,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    ).to(device)

    # Class-weighted loss
    train_labels = [s["label"] for s in train_samples]
    counts = Counter(train_labels)
    tot = len(train_labels)
    raw_weights = [tot / (NUM_CLASSES * max(1, counts[i])) for i in range(NUM_CLASSES)]
    smooth_weights = [math.sqrt(w) for w in raw_weights]
    norm_weights = [w / sum(smooth_weights) * NUM_CLASSES for w in smooth_weights]
    weights_tensor = torch.tensor(norm_weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_loader = DataLoader(
        SequenceFeatureDataset(train_samples, target_seq_len=16, augment=True, strong_aug=strong_aug),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceFeatureDataset(val_samples, target_seq_len=16, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceFeatureDataset(test_samples, target_seq_len=16, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    best_val_metrics = {}
    stagnant_epochs = 0
    t0 = time.time()

    curves = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= max(1, len(train_samples))

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x_dev, y_dev = x.to(device), y.to(device)
                logits = model(x_dev)
                loss = criterion(logits, y_dev)
                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.numpy())

        val_loss /= max(1, len(val_samples))
        val_m = compute_metrics(np.array(val_targets), np.array(val_preds))
        val_macro_f1 = val_m["macro_f1"]
        scheduler.step(val_macro_f1)

        curves["epoch"].append(ep)
        curves["train_loss"].append(round(train_loss, 4))
        curves["val_loss"].append(round(val_loss, 4))
        curves["val_accuracy"].append(val_m["accuracy"])
        curves["val_macro_f1"].append(val_m["macro_f1"])
        curves["val_weighted_f1"].append(val_m["weighted_f1"])

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = ep
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_m
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        if ep % 5 == 0 or ep == epochs or stagnant_epochs >= patience:
            logger.info(
                f"  [{exp_id:>18} | Ep {ep:2d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Acc: {val_m['accuracy']:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} (Best Ep {best_epoch}: {best_val_macro_f1:.4f})"
            )

        if stagnant_epochs >= patience:
            logger.info(f"Early stopping triggered at epoch {ep} (best epoch {best_epoch}).")
            break

    # Save Checkpoint
    ckpt_path = CKPT_DIR / f"{exp_id}_best.pth"
    torch.save(
        {
            "exp_id": exp_id,
            "state_dict": best_state_dict,
            "best_epoch": best_epoch,
            "backbone_type": backbone_type,
            "temporal_type": temporal_type,
            "use_attention": use_attention,
            "classes": CLASSES,
            "val_metrics": best_val_metrics,
        },
        ckpt_path,
    )

    # Save Curves
    curves_path = CURVES_DIR / f"{exp_id}_curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    # Evaluate Best Checkpoint on Held-Out Test Set
    model.load_state_dict(best_state_dict)
    model.eval()

    test_preds, test_targets = [], []
    back_row_preds, back_row_targets = [], []
    video_predictions = defaultdict(lambda: {"true": [], "pred": []})

    with torch.no_grad():
        for x, y, is_br, vids in test_loader:
            x_dev = x.to(device)
            logits = model(x_dev)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = y.numpy()
            is_br_arr = np.array(list(is_br), dtype=bool)

            test_preds.extend(preds)
            test_targets.extend(targets)

            for p_v, t_v, br, vid in zip(preds, targets, is_br_arr, vids):
                if br:
                    back_row_preds.append(p_v)
                    back_row_targets.append(t_v)
                video_predictions[vid]["true"].append(int(t_v))
                video_predictions[vid]["pred"].append(int(p_v))

    test_m = compute_metrics(np.array(test_targets), np.array(test_preds))

    if back_row_targets:
        back_row_m = compute_metrics(np.array(back_row_targets), np.array(back_row_preds))
    else:
        back_row_m = {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    per_video_m = {}
    for vid, data in video_predictions.items():
        v_true = np.array(data["true"])
        v_pred = np.array(data["pred"])
        v_m = compute_metrics(v_true, v_pred)
        per_video_m[vid] = {
            "sequences": len(v_true),
            "accuracy": v_m["accuracy"],
            "macro_f1": v_m["macro_f1"],
        }

    duration = time.time() - t0
    gen_gap = round(best_val_metrics["macro_f1"] - test_m["macro_f1"], 4)

    config = {
        "exp_id": exp_id,
        "backbone_type": backbone_type,
        "temporal_type": temporal_type,
        "use_attention": use_attention,
        "sequence_length": 16,
        "hidden_dim": 256,
        "num_layers": 2,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "strong_aug": strong_aug,
        "best_epoch": best_epoch,
        "total_epochs": len(curves["epoch"]),
        "duration_sec": round(duration, 1),
    }
    if extra_config:
        config.update(extra_config)

    with open(CONFIGS_DIR / f"{exp_id}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    result_payload = {
        "config": config,
        "val_metrics": best_val_metrics,
        "test_metrics": test_m,
        "back_row_metrics": back_row_m,
        "per_video_metrics": per_video_m,
        "generalization_gap": gen_gap,
    }

    metrics_path = METRICS_DIR / f"{exp_id}_results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    logger.info(
        f"Experiment {exp_id} Finished ({duration:.1f}s):\n"
        f"  Val Macro-F1 : {best_val_metrics['macro_f1']:.4f} | Val Accuracy: {best_val_metrics['accuracy']:.4f}\n"
        f"  Test Macro-F1: {test_m['macro_f1']:.4f} | Test Accuracy: {test_m['accuracy']:.4f}\n"
        f"  Gen Gap      : {gen_gap:+.4f} (Val - Test)"
    )

    return result_payload


# ── Main Experiment Runner ──
def main():
    parser = argparse.ArgumentParser(description="TEMPO Behaviour Recognition Ablation & Generalization Suite")
    parser.add_argument("--run-ablation", action="store_true", help="Run full Architecture Ablation Matrix (A through G)")
    parser.add_argument("--run-generalization", action="store_true", help="Run Controlled Generalization Study")
    parser.add_argument("--run-all", action="store_true", help="Run both Ablation and Generalization studies")
    parser.add_argument("--model", type=str, default="", help="Run a specific model (e.g. r18_rnn, r18_gru, r50_bigru_attn)")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of cached features")
    args = parser.parse_args()

    # Create directories
    for p in [RESEARCH_DIR, CACHE_DIR, CKPT_DIR, METRICS_DIR, CURVES_DIR, CONFIGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} | PyTorch version: {torch.__version__}")

    # 1. Load Dataset Index & Sequences
    logger.info("Loading index_repaired.csv and sequences_repaired.json...")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(SEQ_PATH, "r", encoding="utf-8") as f:
        seq_dict = json.load(f)

    train_crops = [r for r in rows if r["split"] == "train"]
    val_crops = [r for r in rows if r["split"] == "val"]
    test_crops = [r for r in rows if r["split"] == "test"]

    # Crop vertical center back-row map (y < 0.35)
    crop_back_row_map = {}
    for r in rows:
        bbox_str = r.get("bbox_norm", "")
        if bbox_str:
            parts = [float(x) for x in bbox_str.split(",")]
            if len(parts) == 4:
                y_center = (parts[1] + parts[3]) / 2.0
                crop_back_row_map[r["relative_path"]] = y_center < 0.35

    all_crop_paths = set()
    for seq_list in seq_dict.values():
        for s in seq_list:
            all_crop_paths.update(s["crop_paths"])
    unique_paths = sorted(all_crop_paths)
    logger.info(f"Unique crop paths across all sequences: {len(unique_paths)}")

    # 2. Build Sequence Sample Builder
    def build_samples(feature_dict):
        train_s, val_s, test_s = [], [], []
        for cls_name, items in seq_dict.items():
            for s in items:
                feats = [feature_dict[p] for p in s["crop_paths"] if p in feature_dict]
                if len(feats) >= 8:
                    is_br = any(crop_back_row_map.get(p, False) for p in s["crop_paths"])
                    sample = {
                        "features": np.stack(feats),
                        "label": CL2I[cls_name],
                        "video_id": s.get("video_id") or s.get("source_video_id", ""),
                        "is_back_row": is_br,
                    }
                    if s["split"] == "train":
                        train_s.append(sample)
                    elif s["split"] == "val":
                        val_s.append(sample)
                    elif s["split"] == "test":
                        test_s.append(sample)
        return train_s, val_s, test_s

    # 3. Load ResNet-18 Spatial Features (Instant if .npy or .pt exists)
    r18_npy = CACHE_DIR / "resnet18_baseline_features.npy"
    r18_pt = CACHE_DIR / "resnet18_baseline_features.pt"
    if r18_npy.exists() or r18_pt.exists():
        logger.info("Found cached ResNet-18 features, loading...")
        r18_backbone = resnet18(weights=None)
        r18_features = get_or_extract_features("resnet18_baseline", r18_backbone, unique_paths, device, force_extract=False)
    else:
        logger.info("Building ResNet-18 baseline spatial backbone...")
        r18_backbone = fine_tune_spatial_backbone("resnet18", train_crops, device, augmentation="baseline")
        r18_features = get_or_extract_features("resnet18_baseline", r18_backbone, unique_paths, device, force_extract=True)

    r18_train, r18_val, r18_test = build_samples(r18_features)
    logger.info(f"ResNet-18 Sequences: Train={len(r18_train)}, Val={len(r18_val)}, Test={len(r18_test)}")

    run_ablation = args.run_ablation or args.run_all or (not args.run_generalization and not args.model)

    # 4. Phase 1: Train Models A, B, C, D (ResNet-18 Family)
    if run_ablation or (args.model and any(m in args.model.lower() for m in ["r18", "exp_a", "exp_b", "exp_c", "exp_d"])):
        print("\n" + "=" * 80)
        print("  STAGE 1A: RESNET-18 ARCHITECTURE ABLATION (Models A through D)")
        print("=" * 80)
        r18_exps = [
            {"exp_id": "exp_A_r18_rnn", "temporal": "RNN", "attn": False},
            {"exp_id": "exp_B_r18_gru", "temporal": "GRU", "attn": False},
            {"exp_id": "exp_C_r18_bigru", "temporal": "BIGRU", "attn": False},
            {"exp_id": "exp_D_r18_bigru_attn", "temporal": "BIGRU", "attn": True},
        ]
        for exp in r18_exps:
            if args.model and (args.model.lower() not in exp["exp_id"].lower() and args.model.lower() not in exp["temporal"].lower()):
                continue
            seed_everything(42)
            train_single_ablation_model(
                exp_id=exp["exp_id"],
                backbone_type="resnet18",
                temporal_type=exp["temporal"],
                use_attention=exp["attn"],
                train_samples=r18_train,
                val_samples=r18_val,
                test_samples=r18_test,
                device=device,
                epochs=30,
                patience=8,
                lr=1e-3,
                batch_size=32,
                strong_aug=False,
                extra_config={"ablation_group": "architecture_ablation"},
            )

    # 5. Phase 2: Load / Extract ResNet-50 Features & Train Models E, F, G
    if run_ablation or (args.model and any(m in args.model.lower() for m in ["r50", "exp_e", "exp_f", "exp_g"])):
        print("\n" + "=" * 80)
        print("  STAGE 1B: RESNET-50 ARCHITECTURE ABLATION (Models E through G)")
        print("=" * 80)
        r50_npy = CACHE_DIR / "resnet50_baseline_features.npy"
        if r50_npy.exists() and not args.force_extract:
            logger.info("Found cached ResNet-50 features, loading...")
            r50_backbone = resnet50(weights=None)
            r50_features = get_or_extract_features("resnet50_baseline", r50_backbone, unique_paths, device, force_extract=False)
        else:
            if PROD_WEIGHTS_PATH.exists():
                logger.info("Loading fine-tuned ResNet-50 backbone from production weights...")
                prod_sd = torch.load(PROD_WEIGHTS_PATH, map_location="cpu", weights_only=False)
                bb_sd = {k.replace("backbone.", ""): v for k, v in prod_sd.items() if k.startswith("backbone.")}
                base_r50 = resnet50(weights=None)
                r50_backbone = nn.Sequential(*list(base_r50.children())[:-1])
                r50_backbone.load_state_dict(bb_sd)
            else:
                logger.info("Fine-tuning ResNet-50 backbone...")
                r50_backbone = fine_tune_spatial_backbone("resnet50", train_crops, device, augmentation="baseline")
            r50_features = get_or_extract_features("resnet50_baseline", r50_backbone, unique_paths, device, force_extract=True)

        r50_train, r50_val, r50_test = build_samples(r50_features)
        logger.info(f"ResNet-50 Sequences: Train={len(r50_train)}, Val={len(r50_val)}, Test={len(r50_test)}")

        r50_exps = [
            {"exp_id": "exp_E_r50_gru", "temporal": "GRU", "attn": False},
            {"exp_id": "exp_F_r50_bigru", "temporal": "BIGRU", "attn": False},
            {"exp_id": "exp_G_r50_bigru_attn", "temporal": "BIGRU", "attn": True},
        ]
        for exp in r50_exps:
            if args.model and (args.model.lower() not in exp["exp_id"].lower() and args.model.lower() not in exp["temporal"].lower()):
                continue
            seed_everything(42)
            train_single_ablation_model(
                exp_id=exp["exp_id"],
                backbone_type="resnet50",
                temporal_type=exp["temporal"],
                use_attention=exp["attn"],
                train_samples=r50_train,
                val_samples=r50_val,
                test_samples=r50_test,
                device=device,
                epochs=30,
                patience=8,
                lr=1e-3,
                batch_size=32,
                strong_aug=False,
                extra_config={"ablation_group": "architecture_ablation"},
            )

    # 5. Controlled Generalization & Fine-Tuning Study
    run_gen = args.run_generalization or args.run_all

    if run_gen:
        print("\n" + "=" * 80)
        print("  STAGE 2: CONTROLLED GENERALIZATION & FINE-TUNING EXPERIMENTS")
        print("=" * 80)

        # Experiment 2.1: Frozen Pretrained Backbone (ImageNet weights, zero fine-tuning)
        r18_frozen_cache = CACHE_DIR / "resnet18_frozen_features.npy"
        if not r18_frozen_cache.exists() or args.force_extract:
            logger.info("Extracting features from frozen ResNet-18 (ImageNet pretrained)...")
            r18_frozen_bb = resnet18(weights=ResNet18_Weights.DEFAULT)
            r18_frozen_feats = get_or_extract_features(
                "resnet18_frozen", r18_frozen_bb, unique_paths, device, force_extract=args.force_extract
            )
        else:
            r18_frozen_feats = get_or_extract_features(
                "resnet18_frozen", None, unique_paths, device, force_extract=False
            )

        r18_frz_tr, r18_frz_val, r18_frz_te = build_samples(r18_frozen_feats)

        seed_everything(42)
        train_single_ablation_model(
            exp_id="exp_gen_r18_frozen_gru",
            backbone_type="resnet18",
            temporal_type="GRU",
            use_attention=False,
            train_samples=r18_frz_tr,
            val_samples=r18_frz_val,
            test_samples=r18_frz_te,
            device=device,
            epochs=30,
            patience=8,
            lr=1e-3,
            strong_aug=False,
            extra_config={"generalization_study": "backbone_strategy", "strategy": "frozen_backbone"},
        )

        # Experiment 2.2: Strong Classroom Augmentation
        r18_strong_cache = CACHE_DIR / "resnet18_strong_features.npy"
        if not r18_strong_cache.exists() or args.force_extract:
            logger.info("Fine-tuning ResNet-18 with strong classroom augmentations...")
            r18_strong_bb = fine_tune_spatial_backbone(
                "resnet18", train_crops, device, augmentation="strong", epochs=3
            )
            r18_strong_feats = get_or_extract_features(
                "resnet18_strong", r18_strong_bb, unique_paths, device, force_extract=args.force_extract
            )
        else:
            r18_strong_feats = get_or_extract_features(
                "resnet18_strong", None, unique_paths, device, force_extract=False
            )

        r18_str_tr, r18_str_val, r18_str_te = build_samples(r18_strong_feats)

        seed_everything(42)
        train_single_ablation_model(
            exp_id="exp_gen_r18_strong_gru",
            backbone_type="resnet18",
            temporal_type="GRU",
            use_attention=False,
            train_samples=r18_str_tr,
            val_samples=r18_str_val,
            test_samples=r18_str_te,
            device=device,
            epochs=30,
            patience=8,
            lr=1e-3,
            strong_aug=True,
            extra_config={"generalization_study": "augmentation", "strategy": "strong_classroom_augmentation"},
        )

        seed_everything(42)
        train_single_ablation_model(
            exp_id="exp_gen_r18_strong_bigru_attn",
            backbone_type="resnet18",
            temporal_type="BIGRU",
            use_attention=True,
            train_samples=r18_str_tr,
            val_samples=r18_str_val,
            test_samples=r18_str_te,
            device=device,
            epochs=30,
            patience=8,
            lr=1e-3,
            strong_aug=True,
            extra_config={"generalization_study": "augmentation", "strategy": "strong_classroom_augmentation"},
        )

        # Experiment 2.3: ResNet-50 with Strong Classroom Augmentation & Lower LR Head
        if "r50_train" not in locals() or r50_train is None:
            r50_npy = CACHE_DIR / "resnet50_baseline_features.npy"
            if r50_npy.exists():
                r50_features = get_or_extract_features(
                    "resnet50_baseline", None, unique_paths, device, force_extract=False
                )
                r50_train, r50_val, r50_test = build_samples(r50_features)
            else:
                logger.warning("ResNet-50 features not found; skipping exp_gen_r50_strong_bigru_attn.")
                r50_train = None

        if r50_train is not None:
            seed_everything(42)
            train_single_ablation_model(
                exp_id="exp_gen_r50_strong_bigru_attn",
                backbone_type="resnet50",
                temporal_type="BIGRU",
                use_attention=True,
                train_samples=r50_train,
                val_samples=r50_val,
                test_samples=r50_test,
                device=device,
                epochs=30,
                patience=8,
                lr=5e-4,  # Lower LR for better convergence on ResNet-50
                strong_aug=True,
                extra_config={"generalization_study": "discriminative_regularization", "strategy": "lower_lr_strong_aug"},
            )

    # 6. Auto-generate comparison summary
    try:
        from scripts.compare_behavior_models import generate_summary
        generate_summary()
    except Exception as e:
        logger.warning(f"Could not generate summary report automatically: {e}")

    logger.info("Training pipeline execution complete.")


if __name__ == "__main__":
    main()
