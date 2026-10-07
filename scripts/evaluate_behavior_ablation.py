#!/usr/bin/env python3
"""
TEMPO — Behaviour Model Ablation & Generalization Evaluator
==========================================================
Evaluates saved ablation checkpoints strictly and reproducibly on:
  1. The 223-sequence Validation Split (14 videos)
  2. The completely frozen 213-sequence Held-Out Test Split (14 unseen videos)

Produces:
  - Overall accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1
  - Per-class Precision, Recall, F1, and Support
  - 5x5 Confusion Matrices
  - Dedicated back-row student metrics
  - 14-video test performance breakdown
  - Generalization gap (Val Macro-F1 - Test Macro-F1)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tempo.ml.eval_ablation")

CLASSES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]
NUM_CLASSES = len(CLASSES)
CL2I = {c: i for i, c in enumerate(CLASSES)}


# ── Import Architecture from train script ──
from scripts.train_behavior_ablation import (
    SequenceFeatureDataset,
    UnifiedAblationModel,
    compute_metrics,
)


def load_dataset_sequences(feature_dict: Dict[str, np.ndarray]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    crop_back_row_map = {}
    for r in rows:
        bbox_str = r.get("bbox_norm", "")
        if bbox_str:
            parts = [float(x) for x in bbox_str.split(",")]
            if len(parts) == 4:
                y_center = (parts[1] + parts[3]) / 2.0
                crop_back_row_map[r["relative_path"]] = y_center < 0.35

    with open(SEQ_PATH, "r", encoding="utf-8") as f:
        seq_dict = json.load(f)

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


def evaluate_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    custom_features_path: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)

    exp_id = ckpt.get("exp_id", ckpt_path.stem.replace("_best", ""))
    backbone_type = ckpt.get("backbone_type", "resnet18")
    temporal_type = ckpt.get("temporal_type", "GRU")
    use_attention = ckpt.get("use_attention", False)

    # Determine which feature cache to load (.npy preferred, .pt fallback)
    crop_paths_file = CACHE_DIR / "crop_paths.json"
    if crop_paths_file.exists():
        with open(crop_paths_file, "r", encoding="utf-8") as fp:
            all_crop_paths = json.load(fp)
    else:
        with open(SEQ_PATH, "r", encoding="utf-8") as fp:
            seq_d = json.load(fp)
        all_crop_paths = sorted(list(set(p for v in seq_d.values() for s in v for p in s["crop_paths"])))

    feat_npy = None
    if custom_features_path and custom_features_path.exists():
        feat_npy = custom_features_path
    elif "frozen" in exp_id.lower() and (CACHE_DIR / f"{backbone_type}_frozen_features.npy").exists():
        feat_npy = CACHE_DIR / f"{backbone_type}_frozen_features.npy"
    elif "strong" in exp_id.lower() and (CACHE_DIR / f"{backbone_type}_strong_features.npy").exists():
        feat_npy = CACHE_DIR / f"{backbone_type}_strong_features.npy"
    elif (CACHE_DIR / f"{backbone_type}_baseline_features.npy").exists():
        feat_npy = CACHE_DIR / f"{backbone_type}_baseline_features.npy"

    if feat_npy and feat_npy.exists():
        mat = np.load(feat_npy)
        feature_dict = {p: mat[i] for i, p in enumerate(all_crop_paths)}
    else:
        feat_pt = CACHE_DIR / f"{backbone_type}_baseline_features.pt"
        if not feat_pt.exists():
            raise FileNotFoundError(f"Neither .npy nor .pt feature cache found for {backbone_type}")
        feature_dict = torch.load(feat_pt, map_location="cpu", weights_only=False)

    train_s, val_s, test_s = load_dataset_sequences(feature_dict)

    model = UnifiedAblationModel(
        backbone_type=backbone_type,
        temporal_type=temporal_type,
        use_attention=use_attention,
        hidden_dim=256,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_loader = DataLoader(
        SequenceFeatureDataset(val_s, target_seq_len=16, augment=False),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceFeatureDataset(test_s, target_seq_len=16, augment=False),
        batch_size=64,
        shuffle=False,
    )

    # 1. Validation Evaluation
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(y.numpy())
    val_m = compute_metrics(np.array(val_targets), np.array(val_preds))

    # 2. Test Set Evaluation
    test_preds, test_targets = [], []
    back_row_preds, back_row_targets = [], []
    video_predictions = defaultdict(lambda: {"true": [], "pred": []})

    with torch.no_grad():
        for x, y, is_br, vids in test_loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            targets = y.numpy()
            is_br_arr = np.array(is_br)

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

    gen_gap = round(val_m["macro_f1"] - test_m["macro_f1"], 4)

    # Print Formatted Evaluation Report
    print("\n" + "=" * 80)
    print(f"  EVALUATION REPORT: {exp_id} ({backbone_type.upper()} + {temporal_type}{'+Attn' if use_attention else ''})")
    print("=" * 80)
    print(f"Validation Performance: Acc={val_m['accuracy']:.4f} | Macro-P={val_m['macro_precision']:.4f} | Macro-R={val_m['macro_recall']:.4f} | Macro-F1={val_m['macro_f1']:.4f} | Weighted-F1={val_m['weighted_f1']:.4f}")
    print(f"Held-Out Test Set:      Acc={test_m['accuracy']:.4f} | Macro-P={test_m['macro_precision']:.4f} | Macro-R={test_m['macro_recall']:.4f} | Macro-F1={test_m['macro_f1']:.4f} | Weighted-F1={test_m['weighted_f1']:.4f}")
    print(f"Generalization Gap:     {gen_gap:+.4f} (Val Macro-F1 - Test Macro-F1)")
    print(f"Back-Row Test Students: Acc={back_row_m['accuracy']:.4f} | Macro-F1={back_row_m['macro_f1']:.4f} (Samples: {len(back_row_targets)})")

    print("\nPER-CLASS TEST BREAKDOWN:")
    print(f"  {'Class Name':<30} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7}")
    print("  " + "-" * 74)
    for c, metrics in test_m["per_class"].items():
        print(f"  {c:<30} | {metrics['precision']:>9.4f} | {metrics['recall']:>9.4f} | {metrics['f1']:>9.4f} | {metrics['support']:>7d}")

    print("\nTEST CONFUSION MATRIX:")
    header = " " * 18 + "".join(f"{c[:6]:>8}" for c in CLASSES)
    print(f"  {header}")
    for i, row in enumerate(test_m["confusion_matrix"]):
        print(f"  {CLASSES[i][:16]:<18}" + "".join(f"{v:>8d}" for v in row))

    result_payload = {
        "exp_id": exp_id,
        "backbone_type": backbone_type,
        "temporal_type": temporal_type,
        "use_attention": use_attention,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "back_row_metrics": back_row_m,
        "per_video_metrics": per_video_m,
        "generalization_gap": gen_gap,
    }

    out_json = METRICS_DIR / f"{exp_id}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate TEMPO behaviour ablation checkpoints")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to specific checkpoint file")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints in checkpoints dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        if not ckpt_p.exists():
            ckpt_p = CKPT_DIR / args.checkpoint
        evaluate_checkpoint(ckpt_p, device)
    elif args.all:
        ckpts = sorted(list(CKPT_DIR.glob("*_best.pth")))
        if not ckpts:
            print("No checkpoints found in", CKPT_DIR)
            return
        for cp in ckpts:
            evaluate_checkpoint(cp, device)
    else:
        print("Please specify --checkpoint <path> or --all")


if __name__ == "__main__":
    main()
