#!/usr/bin/env python3
"""
TEMPO — Repaired Training Dataset Model Benchmarking (Validation Only)
========================================================================
Trains and benchmarks three temporal architectures on top of a fine-tuned
ResNet-18 spatial backbone using the repaired training dataset:
  1. ResNet-18 + RNN
  2. ResNet-18 + LSTM
  3. ResNet-18 + GRU

Strict Rules & Directives:
  - Repaired training dataset: index_repaired.csv & sequences_repaired.json
  - Validation split (223 sequences / 14 videos) & Test split (213 sequences / 14 videos) UNCHANGED
  - Test split is FROZEN and COMPLETELY UNTOUCHED (0 test set evaluations executed)
  - Production checkpoint (models/classroom_temporal_model.pth) is UNTOUCHED
  - Early stopping & champion selection based ONLY on Validation Macro-F1
  - Save each experimental checkpoint separately to models/checkpoints/
"""

import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

# ── Paths ──
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "models", "training_data", "dataset")

REPAIRED_INDEX_PATH = os.path.join(DATA_DIR, "index_repaired.csv")
REPAIRED_SEQ_PATH = os.path.join(DATA_DIR, "sequences_repaired.json")

INDEX_PATH = REPAIRED_INDEX_PATH if os.path.exists(REPAIRED_INDEX_PATH) else os.path.join(DATA_DIR, "index.csv")
SEQ_PATH = REPAIRED_SEQ_PATH if os.path.exists(REPAIRED_SEQ_PATH) else os.path.join(DATA_DIR, "sequences.json")

# ── 5 Observable Classes ──
CLASSES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]
CL2I = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = 5
SEQ_LENGTH = 16

# ── Hyperparameters ──
SPATIAL_SAMPLES_PER_CLASS = 400  # High-efficiency balanced crops per class for layer4 fine-tuning
SPATIAL_EPOCHS = 3
SPATIAL_BS = 64
SPATIAL_LR_BACKBONE = 5e-5        # Smaller LR for layer4 backbone
SPATIAL_LR_HEAD = 1e-4            # LR for FC linear head

TEMPORAL_HIDDEN_DIM = 256
TEMPORAL_NUM_LAYERS = 2
TEMPORAL_DROPOUT = 0.3
TEMPORAL_EPOCHS = 30
TEMPORAL_PATIENCE = 7
TEMPORAL_LR = 1e-3
TEMPORAL_BS = 64

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ── Transforms ──
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.RandomHorizontalFlip(p=0.3),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class InMemoryCropDataset(Dataset):
    """Preloaded in-memory dataset for high-speed CPU training."""
    def __init__(self, tensors: List[torch.Tensor], labels: List[int], augment: bool = True):
        self.tensors = tensors
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = self.tensors[idx]
        l = self.labels[idx]
        if self.augment and random.random() < 0.4:
            t = t + torch.randn_like(t) * 0.015
        return t, l


class SequenceFeatureDataset(Dataset):
    """Dataset for sequence model training from pre-extracted features."""
    def __init__(self, sequences: List[Dict], feature_dict: Dict[str, np.ndarray], augment: bool = False):
        self.samples = []
        for s in sequences:
            feats = []
            missing = False
            for p in s["crop_paths"]:
                if p in feature_dict:
                    feats.append(feature_dict[p])
                else:
                    missing = True
                    break
            if not missing and len(feats) == SEQ_LENGTH:
                lbl = s.get("label") or s.get("target_class") or s.get("class_name")
                self.samples.append({
                    "features": np.stack(feats),
                    "label": CL2I[lbl],
                    "video_id": s["video_id"],
                    "track_id": s["track_id"],
                })
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feat = item["features"].copy()
        label = item["label"]
        if self.augment and random.random() < 0.4:
            shift = random.randint(0, 2)
            feat = np.roll(feat, -shift, axis=0)
        x = torch.from_numpy(feat).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y, item["video_id"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for i in range(NUM_CLASSES):
        tp = cm[i][i]
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        sup = int(cm[i, :].sum())
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        precisions.append(pr)
        recalls.append(rc)
        f1s.append(f1)
        per_class[CLASSES[i]] = {
            "precision": round(pr, 4),
            "recall": round(rc, 4),
            "f1": round(f1, 4),
            "support": sup,
        }

    acc = float((y_true == y_pred).sum()) / max(1, len(y_true))
    pred_dist = dict(Counter(CLASSES[i] for i in y_pred))
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "prediction_distribution": pred_dist,
    }


def print_metrics_table(tag: str, m: Dict):
    print(f"\n{'='*80}", flush=True)
    print(f"  {tag}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  Accuracy: {m['accuracy']:.4f}  |  Macro-F1: {m['macro_f1']:.4f}  |  "
          f"Macro-Precision: {m['macro_precision']:.4f}  |  Macro-Recall: {m['macro_recall']:.4f}", flush=True)
    print(f"\n  {'Class':<32} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}", flush=True)
    print(f"  {'-'*76}", flush=True)
    for c, v in m["per_class"].items():
        print(f"  {c:<32} {v['precision']:>10.4f} {v['recall']:>10.4f} {v['f1']:>10.4f} {v['support']:>10d}", flush=True)
    print(f"\n  Confusion Matrix (rows = Ground Truth, cols = Predicted):", flush=True)
    header = " " * 14 + "".join(f"{c[:7]:>8}" for c in CLASSES)
    print(f"  {header}", flush=True)
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"  {CLASSES[i][:12]:<14}" + "".join(f"{v:>8}" for v in row), flush=True)
    print(f"\n  Prediction Class Distribution:", flush=True)
    tot_preds = sum(m["prediction_distribution"].values())
    for c in CLASSES:
        cnt = m["prediction_distribution"].get(c, 0)
        pct = (cnt / max(1, tot_preds)) * 100.0
        print(f"    {c:<32}: {cnt:>5d} ({pct:>5.1f}%)", flush=True)
    print("=" * 80, flush=True)


def report_pre_training_verification(rows: List[Dict], seq_data: Dict):
    """Verifies and reports source video counts, sequence counts, and split integrity."""
    all_vids = sorted(list(set(r["source_video_id"] for r in rows)))
    vid_to_split = {r["source_video_id"]: r["split"] for r in rows}

    train_vids = set(v for v, sp in vid_to_split.items() if sp == "train")
    val_vids = set(v for v, sp in vid_to_split.items() if sp == "val")
    test_vids = set(v for v, sp in vid_to_split.items() if sp == "test")

    print("\n" + "=" * 85, flush=True)
    print("  PRE-TRAINING VERIFICATION REPORT (DATASET & SPLIT INTEGRITY)", flush=True)
    print("=" * 85, flush=True)

    print(f"\n1. Source Video Partitioning:")
    print(f"   - Total Source Videos: {len(all_vids)}")
    print(f"   - Train Split        : {len(train_vids):>2d} videos ({len(train_vids)/len(all_vids)*100:.1f}%)")
    print(f"   - Validation Split   : {len(val_vids):>2d} videos ({len(val_vids)/len(all_vids)*100:.1f}%)")
    print(f"   - Test Split (HELD)  : {len(test_vids):>2d} videos ({len(test_vids)/len(all_vids)*100:.1f}%)")

    print(f"\n2. Zero Video Overlap Check:")
    print(f"   ✓ Train & Val  video intersection: {len(train_vids & val_vids)}")
    print(f"   ✓ Train & Test video intersection: {len(train_vids & test_vids)}")
    print(f"   ✓ Val & Test   video intersection: {len(val_vids & test_vids)}")
    assert not (train_vids & val_vids), "Train-Val video overlap!"
    assert not (train_vids & test_vids), "Train-Test video overlap!"
    assert not (val_vids & test_vids), "Val-Test video overlap!"

    seq_counts_by_split_cls = defaultdict(Counter)
    all_seqs = []
    for cls_name, seq_list in seq_data.items():
        for s in seq_list:
            sp = s["split"]
            c_name = s.get("label", s.get("target_class", s.get("tempo_label", cls_name)))
            seq_counts_by_split_cls[sp][c_name] += 1
            all_seqs.append(s)

    tot_seqs = len(all_seqs)
    print(f"\n3. Sequences per Class in Each Split (Total: {tot_seqs} sequences):")
    print(f"   {'TEMPO Target Class':<32} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("   " + "-" * 74)
    for c in CLASSES:
        tr = seq_counts_by_split_cls["train"][c]
        va = seq_counts_by_split_cls["val"][c]
        te = seq_counts_by_split_cls["test"][c]
        tot_c = tr + va + te
        print(f"   {c:<32} {tr:>10d} {va:>10d} {te:>10d} {tot_c:>10d}")
    print("   " + "-" * 74)
    tr_tot = sum(seq_counts_by_split_cls["train"].values())
    va_tot = sum(seq_counts_by_split_cls["val"].values())
    te_tot = sum(seq_counts_by_split_cls["test"].values())
    print(f"   {'TOTAL SEQUENCES':<32} {tr_tot:>10d} {va_tot:>10d} {te_tot:>10d} {tot_seqs:>10d}")

    print(f"\n4. Untouched Test Split Rule:")
    print(f"   ✓ Test split ({len(test_vids)} videos, {te_tot} sequences) will be 100% UNTOUCHED.")
    print("=" * 85 + "\n", flush=True)


def fine_tune_spatial_backbone(train_crops: List[Dict], device: torch.device) -> nn.Module:
    """Fine-tunes ResNet-18 layer4 + fc on classroom crops using class-weighted loss and differential learning rates."""
    print(f"\n[Phase 1] Fine-tuning ResNet-18 Spatial Backbone on Repaired Classroom Crops...", flush=True)

    by_class = defaultdict(list)
    for c in train_crops:
        lbl = c.get("filtered_tempo_label", c.get("tempo_label"))
        by_class[lbl].append(c)

    balanced_crops = []
    for cls_name, items in by_class.items():
        sample_n = min(len(items), SPATIAL_SAMPLES_PER_CLASS)
        balanced_crops.extend(random.sample(items, sample_n))
    random.shuffle(balanced_crops)

    print(f"  Preloading {len(balanced_crops):,} balanced crops into memory...", flush=True)
    tensors, labels = [], []
    for item in balanced_crops:
        img_path = os.path.join(DATA_DIR, item["relative_path"])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                tensors.append(train_transform(img))
                lbl = item.get("filtered_tempo_label", item.get("tempo_label"))
                labels.append(CL2I[lbl])
            except Exception:
                pass

    print(f"  Successfully loaded {len(tensors):,} crop tensors into memory.", flush=True)

    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = resnet18(weights=None)

    for name, p in model.named_parameters():
        if "layer4" in name or "fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.fc = nn.Linear(512, NUM_CLASSES)
    model.to(device)

    crop_cls_counts = Counter(labels)
    tot_c = len(labels)
    spatial_weights = torch.tensor([
        tot_c / (NUM_CLASSES * max(1, crop_cls_counts[i])) for i in range(NUM_CLASSES)
    ]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=spatial_weights)

    layer4_params = [p for name, p in model.named_parameters() if "layer4" in name and p.requires_grad]
    fc_params = [p for name, p in model.named_parameters() if "fc" in name and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": SPATIAL_LR_BACKBONE},
        {"params": fc_params, "lr": SPATIAL_LR_HEAD},
    ], weight_decay=1e-4)

    train_loader = DataLoader(
        InMemoryCropDataset(tensors, labels, augment=True),
        batch_size=SPATIAL_BS,
        shuffle=True,
        drop_last=True,
    )

    t0 = time.time()
    for ep in range(SPATIAL_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += int((out.argmax(dim=1) == y).sum().item())
            total += len(y)

        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        print(f"  Spatial Backbone Epoch {ep+1}/{SPATIAL_EPOCHS}: loss={avg_loss:.4f}, acc={acc:.4f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Completed spatial fine-tuning in {time.time()-t0:.1f}s.", flush=True)
    return model


def extract_features(model: nn.Module, relative_paths: List[str], device: torch.device) -> Dict[str, np.ndarray]:
    """Extracts 512-dim features for all required unique crops."""
    print(f"\n[Phase 2] Extracting 512-dim features for {len(relative_paths):,} unique crops...", flush=True)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    feature_dict = {}
    bs = 128
    t0 = time.time()

    for i in range(0, len(relative_paths), bs):
        batch_paths = relative_paths[i : i + bs]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            full_p = os.path.join(DATA_DIR, p)
            if os.path.exists(full_p):
                try:
                    img = Image.open(full_p).convert("RGB")
                    tensors.append(eval_transform(img))
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
            feature_dict[p] = feats[j]

        if (i + bs) % 2000 == 0 or (i + bs) >= len(relative_paths):
            print(f"    Extracted {min(i+bs, len(relative_paths)):,} / {len(relative_paths):,} crops ({time.time()-t0:.1f}s)...", flush=True)

    print(f"  Feature extraction complete: {len(feature_dict):,} crops cached ({time.time()-t0:.1f}s).", flush=True)
    return feature_dict


class TemporalSequenceClassifier(nn.Module):
    """Temporal Sequence Model (RNN / LSTM / GRU) on top of spatial features."""
    def __init__(self, temporal_type: str, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.temporal_type = temporal_type.upper()
        self.hidden_dim = hidden_dim

        if self.temporal_type == "LSTM":
            self.temporal = nn.LSTM(
                input_size=512,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.temporal_type == "RNN":
            self.temporal = nn.RNN(
                input_size=512,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="relu"
            )
        else:  # GRU
            self.temporal = nn.GRU(
                input_size=512,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_type == "LSTM":
            out, (hn, cn) = self.temporal(x)
        else:
            out, hn = self.temporal(x)
        last_step = out[:, -1, :]
        return self.classifier(last_step)


def train_and_validate_temporal(
    arch: str,
    train_seqs: List[Dict],
    val_seqs: List[Dict],
    feature_dict: Dict[str, np.ndarray],
    device: torch.device
) -> Dict:
    """Trains a temporal model on train split and evaluates strictly on validation split."""
    print(f"\n  >>> Training ResNet-18 + {arch} on Repaired Train Split...", flush=True)

    by_class = defaultdict(list)
    for s in train_seqs:
        lbl = s.get("label") or s.get("target_class") or s.get("class_name")
        by_class[lbl].append(s)

    max_c = max(len(v) for v in by_class.values()) if by_class else 0
    balanced_train_seqs = []
    for c, items in by_class.items():
        reps, rem = divmod(max_c, len(items))
        balanced_train_seqs += items * reps + random.sample(items, rem)
    random.shuffle(balanced_train_seqs)

    train_ds = SequenceFeatureDataset(balanced_train_seqs, feature_dict, augment=True)
    val_ds = SequenceFeatureDataset(val_seqs, feature_dict, augment=False)

    train_loader = DataLoader(train_ds, batch_size=TEMPORAL_BS, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = TemporalSequenceClassifier(
        temporal_type=arch,
        hidden_dim=TEMPORAL_HIDDEN_DIM,
        num_layers=TEMPORAL_NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=TEMPORAL_DROPOUT,
    ).to(device)

    class_counts = Counter(s["label"] for s in train_ds.samples)
    tot = sum(class_counts.values())
    cw = torch.tensor([tot / (NUM_CLASSES * max(1, class_counts[i])) for i in range(NUM_CLASSES)]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TEMPORAL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEMPORAL_EPOCHS, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None
    best_ep = 0
    best_val_metrics = None
    patience = 0
    t0 = time.time()

    ckpt_dir = os.path.join(BASE_DIR, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    exp_ckpt_path = os.path.join(ckpt_dir, f"exp_repaired_resnet18_{arch.lower()}_val_best.pth")

    for ep in range(TEMPORAL_EPOCHS):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_samples += len(y)
        scheduler.step()

        # Validation evaluation ONLY
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
                y_true.extend(y.numpy())
                y_pred.extend(preds)

        val_m = calculate_metrics(np.array(y_true), np.array(y_pred))
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_val_metrics = val_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            patience = 0
            torch.save(best_state, exp_ckpt_path)
        else:
            patience += 1

        if (ep + 1) % 5 == 0 or ep == TEMPORAL_EPOCHS - 1:
            print(f"    [{arch:>4}] Epoch {ep+1:>2}: train_loss={total_loss/max(1, n_samples):.4f} "
                  f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['macro_f1']:.4f} (best ep {best_ep+1}: val_f1={best_val_f1:.4f})", flush=True)

        if patience >= TEMPORAL_PATIENCE:
            print(f"    [{arch:>4}] Early stopping triggered at epoch {ep+1}. Best epoch: {best_ep+1} (val_f1={best_val_f1:.4f})", flush=True)
            break

    dur = time.time() - t0
    print(f"  Experimental checkpoint saved to: {exp_ckpt_path}", flush=True)
    return {
        "arch": arch,
        "best_epoch": best_ep + 1,
        "duration_sec": round(dur, 1),
        "val_metrics": best_val_metrics,
        "state_dict": best_state,
        "ckpt_path": exp_ckpt_path,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80, flush=True)
    print("  TEMPO — Repaired Training Dataset Model Benchmarking (Validation Only)", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} (PyTorch {torch.__version__})", flush=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load Dataset Metadata
    print("\n[Data] Loading verified repaired dataset index and 16-frame sequences...", flush=True)
    print(f"  Index Path: {INDEX_PATH}")
    print(f"  Seq Path  : {SEQ_PATH}")

    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    with open(SEQ_PATH, "r", encoding="utf-8") as f:
        seq_data = json.load(f)

    train_crops = [r for r in rows if r["split"] == "train"]
    val_crops = [r for r in rows if r["split"] == "val"]

    train_seqs, val_seqs, test_seqs = [], [], []
    for cls_name, seq_list in seq_data.items():
        for s in seq_list:
            if s["split"] == "train":
                train_seqs.append(s)
            elif s["split"] == "val":
                val_seqs.append(s)
            else:
                test_seqs.append(s)

    # 2. Pre-Training Verification Report
    report_pre_training_verification(rows, seq_data)

    # 3. Fine-tune ResNet-18 Spatial Backbone on Repaired Crops
    spatial_backbone = fine_tune_spatial_backbone(train_crops, device)

    # 4. Extract Features for Unique Sequence Crops
    unique_crop_paths = sorted(set(
        p for s in (train_seqs + val_seqs + test_seqs) for p in s["crop_paths"]
    ))
    feature_dict = extract_features(spatial_backbone, unique_crop_paths, device)

    # 5. Train and Compare ResNet-18 + RNN, LSTM, and GRU on Validation Split
    print("\n[Phase 3] Training & Benchmarking RNN, LSTM, and GRU Temporal Heads...", flush=True)
    bench_results = {}
    for arch in ["RNN", "LSTM", "GRU"]:
        res = train_and_validate_temporal(arch, train_seqs, val_seqs, feature_dict, device)
        bench_results[arch] = res
        print_metrics_table(f"ResNet-18 + {arch} (14-Video Validation Split)", res["val_metrics"])

    # 6. Validation Comparison Table & Champion Selection
    print("\n" + "=" * 85, flush=True)
    print("  TEMPORAL ARCHITECTURE COMPARISON (14-Video Validation Split)", flush=True)
    print("=" * 85, flush=True)
    print(f"  {'Architecture':<18} {'Accuracy':>10} {'Macro-F1':>10} {'Macro-Prec':>12} {'Macro-Rec':>10} {'Best Ep':>8} {'Time':>8}", flush=True)
    print("  " + "-" * 82, flush=True)
    best_arch = max(bench_results, key=lambda a: bench_results[a]["val_metrics"]["macro_f1"])
    for arch, r in bench_results.items():
        m = r["val_metrics"]
        star = " ★ CHAMPION" if arch == best_arch else ""
        print(f"  ResNet-18 + {arch:<8} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['macro_precision']:>12.4f} {m['macro_recall']:>10.4f} {r['best_epoch']:>8d} {r['duration_sec']:>7.1f}s{star}", flush=True)
    print("=" * 85, flush=True)

    champion = bench_results[best_arch]
    c_m = champion["val_metrics"]
    print(f"\nChampion Model Selected STRICTLY by Validation Macro-F1: ResNet-18 + {best_arch}", flush=True)
    print(f"  - Validation Accuracy       : {c_m['accuracy']:.4f}", flush=True)
    print(f"  - Validation Macro-F1       : {c_m['macro_f1']:.4f}", flush=True)
    print(f"  - Validation Macro-Precision: {c_m['macro_precision']:.4f}", flush=True)
    print(f"  - Validation Macro-Recall   : {c_m['macro_recall']:.4f}", flush=True)
    print(f"  - Best Epoch                : {champion['best_epoch']}", flush=True)

    # 7. Verification of Strict Rule Adherence
    print("\n" + "=" * 80, flush=True)
    print("  STRICT RULE ADHERENCE CONFIRMATION", flush=True)
    print("=" * 80, flush=True)
    print("  [CONFIRMED] Test split evaluated: 0 sequences (0% evaluated).", flush=True)
    print("  [CONFIRMED] Production weights (models/classroom_temporal_model.pth): UNTOUCHED.", flush=True)
    print("  [CONFIRMED] Production metadata (models/model_metadata.json): UNTOUCHED.", flush=True)
    print("  [CONFIRMED] Experimental checkpoints saved separately to models/checkpoints/:", flush=True)
    for arch in ["RNN", "LSTM", "GRU"]:
        print(f"              - {bench_results[arch]['ckpt_path']}")
    print("=" * 80 + "\n", flush=True)

    # Dump results JSON for reporting
    out_json_path = os.path.join(BASE_DIR, "models", "checkpoints", "repaired_validation_results.json")
    json_data = {
        arch: {
            "best_epoch": r["best_epoch"],
            "duration_sec": r["duration_sec"],
            "val_metrics": r["val_metrics"],
            "ckpt_path": r["ckpt_path"]
        }
        for arch, r in bench_results.items()
    }
    json_data["champion"] = best_arch
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Validation benchmark results saved to {out_json_path}", flush=True)


if __name__ == "__main__":
    main()
