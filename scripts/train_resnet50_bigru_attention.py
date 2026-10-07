#!/usr/bin/env python3
"""
TEMPO — ResNet-50 + BiGRU + Temporal Attention Training & Ablation Pipeline
===========================================================================
Upgrades behaviour recognition from ResNet-18 to ResNet-50 with:
  1. Spatial Backbone: ResNet-50 (2048-dim embedding)
  2. Feature Projection + LayerNorm: Linear(2048, 512) -> LayerNorm -> GELU -> Dropout
  3. Temporal Sequence Modeler: 2-layer Bidirectional GRU (hidden=256 -> 512 output)
  4. Temporal Attention: Additive self-attention context aggregation over T frames
  5. Classifier Head: Linear(512, 128) -> LayerNorm -> ReLU -> Linear(128, 5)

Enforces:
  - Clean dataset audit (zero video/session leakage across splits)
  - Class-weighted CrossEntropyLoss
  - Progressive fine-tuning (head first, then layer4 with discriminative LRs)
  - Safe temporal augmentation & temporal consistency regularization
  - Validation-only sequence length selection (8, 16, 24, 32)
  - Dedicated back-row student classification evaluation
  - Single controlled test set evaluation vs ResNet-18 baseline
"""

import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

# ── Setup Project Root ──
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.ml.model import (
    CLASS_NAMES,
    NUM_CLASSES,
    ResNet50BiGRUTemporalModel,
    TemporalAttention,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tempo.ml.train_resnet50")

# ── Paths ──
DATA_DIR = BASE_DIR / "models" / "training_data" / "dataset"
INDEX_PATH = DATA_DIR / "index_repaired.csv"
SEQ_PATH = DATA_DIR / "sequences_repaired.json"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
PROD_WEIGHTS_PATH = BASE_DIR / "models" / "classroom_temporal_model.pth"
PROD_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

CL2I = {c: i for i, c in enumerate(CLASS_NAMES)}

# ── Normalization & Transforms ──
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

spatial_train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.RandomHorizontalFlip(p=0.3),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

spatial_eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


# ── Dataset Classes ──
class CropDataset(Dataset):
    """In-memory crop tensor dataset for progressive spatial fine-tuning."""

    def __init__(self, tensors: List[torch.Tensor], labels: List[int], augment: bool = True):
        self.tensors = tensors
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = self.tensors[idx]
        lbl = self.labels[idx]
        if self.augment and random.random() < 0.35:
            t = t + torch.randn_like(t) * 0.015
        return t, lbl


class SequenceDataset(Dataset):
    """Sequence dataset from pre-extracted 2048-dim ResNet-50 features."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        target_seq_len: int = 16,
        augment: bool = False,
    ):
        self.samples = samples
        self.target_seq_len = target_seq_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feats = item["features"]  # shape (T_orig, 2048)
        T_orig = feats.shape[0]

        # Adapt sequence length to target_seq_len
        if T_orig == self.target_seq_len:
            seq = feats.copy()
        elif T_orig > self.target_seq_len:
            # Subsample evenly
            indices = np.linspace(0, T_orig - 1, self.target_seq_len, dtype=int)
            seq = feats[indices].copy()
        else:
            # Repeat or pad
            pad_count = self.target_seq_len - T_orig
            repeat_pad = np.repeat(feats[-1:], pad_count, axis=0)
            seq = np.concatenate([feats, repeat_pad], axis=0)

        # Safe temporal augmentation
        if self.augment:
            # Feature noise
            if random.random() < 0.40:
                seq = seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32)
            # Frame temporal jitter / dropout (replace 1 frame with adjacent without altering label semantics)
            if self.target_seq_len > 4 and random.random() < 0.30:
                j_idx = random.randint(1, self.target_seq_len - 1)
                seq[j_idx] = seq[j_idx - 1]

        x = torch.from_numpy(seq).float()
        y = torch.tensor(item["label"], dtype=torch.long)
        return x, y, item.get("is_back_row", False), item.get("video_id", "")


# ── Metric Calculation ──
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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

        per_class[CLASS_NAMES[i]] = {
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

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_pr, 4),
        "macro_recall": round(macro_rc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ── Temporal Consistency Regularizer ──
def temporal_consistency_loss(gru_out: torch.Tensor) -> torch.Tensor:
    """
    Penalizes erratic representation changes between consecutive frames.
    L_consistency = mean ||h_{t+1} - h_t||^2
    """
    diff = gru_out[:, 1:, :] - gru_out[:, :-1, :]
    return torch.mean(torch.sum(diff ** 2, dim=-1))


# ── Progressive ResNet-50 Fine-Tuning ──
def fine_tune_spatial_backbone(
    train_crops: List[Dict[str, Any]],
    device: torch.device,
    samples_per_class: int = 350,
) -> nn.Module:
    """
    Progressively fine-tunes ResNet-50:
      Phase 1: Freeze layers 1-4, train linear projection head.
      Phase 2: Unfreeze layer4 with smaller discriminative learning rate.
    """
    logger.info("Initializing Pretrained ResNet-50 spatial backbone...")
    base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Balanced crop sampling for spatial adaptation
    by_class = defaultdict(list)
    for c in train_crops:
        by_class[c["tempo_label"]].append(c)

    selected_crops = []
    for cls_name, items in by_class.items():
        sample_k = min(len(items), samples_per_class)
        selected_crops.extend(random.sample(items, sample_k))
    random.shuffle(selected_crops)

    logger.info(f"Loading {len(selected_crops)} balanced training crops into memory...")
    tensors, labels = [], []
    for c in selected_crops:
        full_p = DATA_DIR / c["relative_path"]
        if full_p.exists():
            try:
                img = Image.open(full_p).convert("RGB")
                tensors.append(spatial_train_transform(img))
                labels.append(CL2I[c["tempo_label"]])
            except Exception:
                pass

    if len(tensors) == 0:
        logger.warning("No image crops found on disk! Using base pretrained ResNet-50.")
        return base_model

    # Replace FC head with 5-class head for fine-tuning
    base_model.fc = nn.Linear(base_model.fc.in_features, NUM_CLASSES)
    base_model.to(device)

    # Class weights
    counts = Counter(labels)
    tot = len(labels)
    weights = torch.tensor([
        tot / (NUM_CLASSES * max(1, counts[i])) for i in range(NUM_CLASSES)
    ]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(
        CropDataset(tensors, labels, augment=True),
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )

    # Phase 1: Train Head only (freeze all backbone)
    logger.info("Spatial Phase 1: Training FC head (backbone frozen)...")
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.fc.parameters():
        param.requires_grad = True

    opt_head = torch.optim.AdamW(base_model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    base_model.train()
    for ep in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_head.zero_grad()
            out = base_model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_head.step()

    # Phase 2: Unfreeze layer4 with discriminative learning rates
    logger.info("Spatial Phase 2: Unfreezing layer4 with discriminative learning rate (lr=2e-5)...")
    for name, param in base_model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    opt_full = torch.optim.AdamW([
        {"params": [p for n, p in base_model.named_parameters() if "layer4" in n], "lr": 2e-5},
        {"params": base_model.fc.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)

    for ep in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_full.zero_grad()
            out = base_model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_full.step()

    logger.info("Spatial fine-tuning complete.")
    return base_model


# ── Spatial Feature Extractor & Cacher ──
def extract_resnet50_features(
    model: nn.Module,
    relative_paths: List[str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Extract 2048-dim features up to avgpool from ResNet-50."""
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    feature_dict = {}
    batch_size = 64

    logger.info(f"Extracting ResNet-50 features for {len(relative_paths)} unique crops...")
    for i in range(0, len(relative_paths), batch_size):
        batch_paths = relative_paths[i : i + batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            full_p = DATA_DIR / p
            if full_p.exists():
                try:
                    img = Image.open(full_p).convert("RGB")
                    tensors.append(spatial_eval_transform(img))
                    valid_paths.append(p)
                except Exception:
                    pass

        if not tensors:
            continue

        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch_t)
            feats = torch.flatten(feats, 1).cpu().numpy()

        for j, p in enumerate(valid_paths):
            feature_dict[p] = feats[j].astype(np.float32)

    logger.info(f"Successfully extracted {len(feature_dict)} feature vectors.")
    return feature_dict


# ── Training Loop for Temporal BiGRU + Attention Model ──
def train_temporal_model(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    target_seq_len: int,
    device: torch.device,
    epochs: int = 35,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    consistency_weight: float = 1e-4,
) -> Tuple[ResNet50BiGRUTemporalModel, Dict[str, Any]]:
    """
    Trains the feature projection + BiGRU + Temporal Attention + Classifier.
    Uses validation Macro-F1 exclusively for early stopping and checkpoint selection.
    """
    model = ResNet50BiGRUTemporalModel(
        hidden_dim=256,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        use_pretrained_backbone=False,
    ).to(device)

    # Class-weighted loss
    train_labels = [s["label"] for s in train_samples]
    counts = Counter(train_labels)
    tot = len(train_labels)
    # Smoothed inverse-frequency weights
    raw_weights = [tot / (NUM_CLASSES * max(1, counts[i])) for i in range(NUM_CLASSES)]
    smooth_weights = [math.sqrt(w) for w in raw_weights]
    norm_weights = [w / sum(smooth_weights) * NUM_CLASSES for w in smooth_weights]
    weights_tensor = torch.tensor(norm_weights).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.projection.parameters(), "lr": lr},
            {"params": model.bigru.parameters(), "lr": lr},
            {"params": model.attention.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_loader = DataLoader(
        SequenceDataset(train_samples, target_seq_len=target_seq_len, augment=True),
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(val_samples, target_seq_len=target_seq_len, augment=False),
        batch_size=32,
        shuffle=False,
    )

    best_val_macro_f1 = -1.0
    best_state_dict = None
    best_val_metrics = {}
    stagnant_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            proj = model.projection(x)
            gru_out, _ = model.bigru(proj)
            context, _ = model.attention(gru_out)
            logits = model.classifier(context)

            ce_loss = criterion(logits, y)
            cons_loss = temporal_consistency_loss(gru_out)
            total_loss = ce_loss + consistency_weight * cons_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            train_loss += total_loss.item() * len(y)

        train_loss /= len(train_samples)

        # Validation Pass
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device)
                logits = model.forward_features(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.numpy())

        val_metrics = compute_classification_metrics(np.array(val_targets), np.array(val_preds))
        val_macro_f1 = val_metrics["macro_f1"]
        scheduler.step(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        if ep % 5 == 0 or ep == epochs or stagnant_epochs >= patience:
            logger.info(
                f"[Seq {target_seq_len:2d} | Ep {ep:2d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} (Best: {best_val_macro_f1:.4f})"
            )

        if stagnant_epochs >= patience:
            logger.info(f"Early stopping triggered at epoch {ep} for sequence length {target_seq_len}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_val_metrics


# ── Full Pipeline Execution ──
def main():
    print("=" * 85)
    print("  TEMPO — ResNet-50 + BiGRU + Temporal Attention Training & Evaluation Pipeline")
    print("=" * 85)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} (PyTorch {torch.__version__})")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Audit
    logger.info("Auditing dataset and loading index & sequences...")
    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(SEQ_PATH, "r", encoding="utf-8") as f:
        seq_dict = json.load(f)

    train_crops = [r for r in rows if r["split"] == "train"]
    val_crops = [r for r in rows if r["split"] == "val"]
    test_crops = [r for r in rows if r["split"] == "test"]

    train_vids = set(r["source_video_id"] for r in train_crops)
    val_vids = set(r["source_video_id"] for r in val_crops)
    test_vids = set(r["source_video_id"] for r in test_crops)

    logger.info(
        f"Crops: Train={len(train_crops)}, Val={len(val_crops)}, Test={len(test_crops)} | "
        f"Videos: Train={len(train_vids)}, Val={len(val_vids)}, Test={len(test_vids)}"
    )
    assert len(train_vids & val_vids) == 0, "Leakage between train and val!"
    assert len(train_vids & test_vids) == 0, "Leakage between train and test!"
    assert len(val_vids & test_vids) == 0, "Leakage between val and test!"
    logger.info("-> Zero video/session leakage strictly verified across all splits.")

    # 2. Progressive Fine-Tuning of Spatial Backbone
    finetuned_backbone = fine_tune_spatial_backbone(train_crops, device)

    # 3. Extract & Cache ResNet-50 Features
    all_needed_paths = set()
    for seq_list in seq_dict.values():
        for s in seq_list:
            all_needed_paths.update(s["crop_paths"])

    features_dict = extract_resnet50_features(finetuned_backbone, sorted(all_needed_paths), device)

    # 4. Construct Sequence Samples
    train_seqs, val_seqs, test_seqs = [], [], []

    # Map crop row to back-row flag if vertical center is in upper region (y < 0.35)
    crop_back_row_map = {}
    for r in rows:
        bbox_str = r.get("bbox_norm", "")
        if bbox_str:
            parts = [float(x) for x in bbox_str.split(",")]
            if len(parts) == 4:
                # y_center = (y1 + y2) / 2
                y_center = (parts[1] + parts[3]) / 2.0
                crop_back_row_map[r["relative_path"]] = y_center < 0.35

    for cls_name, items in seq_dict.items():
        for s in items:
            feats = [features_dict[p] for p in s["crop_paths"] if p in features_dict]
            if len(feats) >= 8:
                is_br = any(crop_back_row_map.get(p, False) for p in s["crop_paths"])
                sample = {
                    "features": np.stack(feats),
                    "label": CL2I[cls_name],
                    "video_id": s.get("video_id") or s.get("source_video_id", ""),
                    "is_back_row": is_br,
                }
                if s["split"] == "train":
                    train_seqs.append(sample)
                elif s["split"] == "val":
                    val_seqs.append(sample)
                elif s["split"] == "test":
                    test_seqs.append(sample)

    logger.info(
        f"Valid Sequence Samples: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}"
    )

    # 5. Sequence Length Experiments (8, 16, 24, 32) on Validation Set
    print("\n" + "=" * 85)
    print("  SEQUENCE LENGTH EXPERIMENTS (Selection guided ONLY by Validation Macro-F1)")
    print("=" * 85)

    seq_experiment_results = {}
    best_seq_len = 16
    best_seq_macro_f1 = -1.0
    models_by_seq = {}

    for t_len in [8, 16, 24, 32]:
        logger.info(f"\n--- Training ResNet-50 + BiGRU + Attention with sequence length T={t_len} ---")
        model_t, val_res = train_temporal_model(
            train_samples=train_seqs,
            val_samples=val_seqs,
            target_seq_len=t_len,
            device=device,
            epochs=25,
            patience=6,
        )
        seq_experiment_results[t_len] = val_res
        models_by_seq[t_len] = model_t

        print(
            f"SeqLen T={t_len:2d} -> Val Accuracy: {val_res['accuracy']:.4f} | "
            f"Val Macro-F1: {val_res['macro_f1']:.4f} | "
            f"Val Weighted-F1: {val_res['weighted_f1']:.4f}"
        )

        if val_res["macro_f1"] > best_seq_macro_f1:
            best_seq_macro_f1 = val_res["macro_f1"]
            best_seq_len = t_len

    print("\n" + "-" * 85)
    print(f"CHAMPION SEQUENCE LENGTH SELECTED: T={best_seq_len} (Val Macro-F1 = {best_seq_macro_f1:.4f})")
    print("-" * 85)

    champion_model = models_by_seq[best_seq_len]

    # Save champion temporal checkpoint
    ckpt_path = CHECKPOINTS_DIR / "resnet50_bigru_attention_champion.pth"
    torch.save(
        {
            "state_dict": champion_model.state_dict(),
            "sequence_length": best_seq_len,
            "hidden_dim": 256,
            "num_layers": 2,
            "classes": CLASS_NAMES,
            "val_metrics": seq_experiment_results[best_seq_len],
        },
        ckpt_path,
    )
    logger.info(f"Saved champion checkpoint to {ckpt_path}")

    # 6. Single Evaluation on the Untouched Frozen Test Set
    print("\n" + "=" * 85)
    print("  CONTROLLED EVALUATION ON UNTOUCHED TEST SET (ResNet-18 vs ResNet-50)")
    print("=" * 85)

    test_loader = DataLoader(
        SequenceDataset(test_seqs, target_seq_len=best_seq_len, augment=False),
        batch_size=32,
        shuffle=False,
    )

    champion_model.eval()
    test_preds, test_targets = [], []
    back_row_preds, back_row_targets = [], []

    with torch.no_grad():
        for x, y, is_br, _ in test_loader:
            x = x.to(device)
            logits = champion_model.forward_features(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = y.numpy()
            is_br_arr = np.array(is_br)

            test_preds.extend(preds)
            test_targets.extend(targets)

            # Back-row separate evaluation
            for p_val, t_val, br_flag in zip(preds, targets, is_br_arr):
                if br_flag:
                    back_row_preds.append(p_val)
                    back_row_targets.append(t_val)

    test_metrics = compute_classification_metrics(np.array(test_targets), np.array(test_preds))

    # Dedicated Back-row metrics
    if back_row_targets:
        back_row_metrics = compute_classification_metrics(
            np.array(back_row_targets), np.array(back_row_preds)
        )
    else:
        back_row_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    # 7. Print Comparative Report
    print("\n" + "-" * 85)
    print("TEST SET METRICS COMPARISON:")
    print(f"ResNet-18 Baseline:  Acc: 0.5915 | Macro-P: 0.4069 | Macro-R: 0.4573 | Macro-F1: 0.3929 | Weighted-F1: 0.5780")
    print(
        f"ResNet-50 + BiGRU:   Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-P: {test_metrics['macro_precision']:.4f} | "
        f"Macro-R: {test_metrics['macro_recall']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Weighted-F1: {test_metrics['weighted_f1']:.4f}"
    )
    print("-" * 85)

    print("\nPER-CLASS TEST SET BREAKDOWN (ResNet-50 + BiGRU + Attention):")
    for cls_name, m in test_metrics["per_class"].items():
        print(
            f"  {cls_name:<30} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | Support: {m['support']:3d}"
        )

    print("\nDEDICATED BACK-ROW STUDENT BEHAVIOUR RECOGNITION:")
    print(
        f"  Back-Row Samples: {len(back_row_targets)} | Accuracy: {back_row_metrics['accuracy']:.4f} | "
        f"Macro-F1: {back_row_metrics['macro_f1']:.4f}"
    )

    print("\nCONFUSION MATRIX:")
    cm = np.array(test_metrics["confusion_matrix"])
    print(f"             " + " ".join(f"{c[:6]:>7}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:12]:<12} " + " ".join(f"{v:>7d}" for v in row))

    # 8. Save Production Weights & Update model_metadata.json
    # Combine full end-to-end model with spatial backbone for production deployment
    logger.info("Exporting complete model to production weights path...")
    full_prod_model = ResNet50BiGRUTemporalModel(
        hidden_dim=256,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_pretrained_backbone=False,
    )
    # Copy backbone from finetuned_backbone and temporal layers from champion_model
    full_prod_model.backbone.load_state_dict(
        nn.Sequential(*list(finetuned_backbone.children())[:-1]).state_dict()
    )
    full_prod_model.projection.load_state_dict(champion_model.projection.state_dict())
    full_prod_model.bigru.load_state_dict(champion_model.bigru.state_dict())
    full_prod_model.attention.load_state_dict(champion_model.attention.state_dict())
    full_prod_model.classifier.load_state_dict(champion_model.classifier.state_dict())

    torch.save(full_prod_model.state_dict(), PROD_WEIGHTS_PATH)
    logger.info(f"Saved production weights to {PROD_WEIGHTS_PATH}")

    # Update metadata
    prod_meta = {
        "model_version": "v3.0.0-resnet50-bigru-attention",
        "spatial_backbone": "resnet50",
        "temporal_model_type": "BiGRU",
        "hidden_dim": 256,
        "num_layers": 2,
        "sequence_length": best_seq_len,
        "sampling_fps": 2.0,
        "input_size": [3, 224, 224],
        "classes": CLASS_NAMES,
        "class_mapping": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "normalization": {
            "mean": NORM_MEAN,
            "std": NORM_STD,
        },
        "metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
        "back_row_metrics": {
            "samples": len(back_row_targets),
            "accuracy": back_row_metrics["accuracy"],
            "macro_f1": back_row_metrics["macro_f1"],
        },
        "validation_sequence_ablation": {
            str(k): {
                "accuracy": v["accuracy"],
                "macro_f1": v["macro_f1"],
                "weighted_f1": v["weighted_f1"],
            }
            for k, v in seq_experiment_results.items()
        },
        "baseline_comparison": {
            "resnet18_baseline": {
                "accuracy": 0.5915,
                "macro_f1": 0.3929,
                "weighted_f1": 0.5780,
            },
            "resnet50_bigru_attention": {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
            },
            "delta_macro_f1": round(test_metrics["macro_f1"] - 0.3929, 4),
            "delta_accuracy": round(test_metrics["accuracy"] - 0.5915, 4),
        },
    }

    with open(PROD_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(prod_meta, f, indent=2)
    logger.info(f"Updated {PROD_METADATA_PATH}")

    # Also save detailed JSON report
    report_path = CHECKPOINTS_DIR / "resnet50_ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_metrics": test_metrics,
            "back_row_metrics": back_row_metrics,
            "seq_ablation": seq_experiment_results,
            "champion_seq_len": best_seq_len,
        }, f, indent=2)
    logger.info(f"Saved full ablation report to {report_path}")

    print("\n==================================================================")
    print("TRAINING, ABLATION & EVALUATION PIPELINE SUCCESSFULLY COMPLETED.")
    print("==================================================================")


if __name__ == "__main__":
    main()
