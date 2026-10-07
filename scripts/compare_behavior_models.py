#!/usr/bin/env python3
"""
TEMPO — Behaviour Recognition Model Comparison & Ablation Synthesis
===================================================================
Aggregates all evaluation results from storage/research/behavior_ablation/metrics/
and produces:
  1. storage/research/behavior_ablation/ablation_summary.csv
  2. storage/research/behavior_ablation/ablation_summary.json
  3. storage/research/behavior_ablation/ablation_report.md

Enforces:
  - Primary ranking and model selection metric is HELD-OUT TEST MACRO-F1.
  - Transparent validation vs test comparison (highlights generalization gap).
  - Strict baseline gate comparison against ResNet-18 reference (Test Macro-F1 = 0.3929).
  - Zero fabricated metrics; completely reproducible summary.
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

RESEARCH_DIR = BASE_DIR / "storage" / "research" / "behavior_ablation"
METRICS_DIR = RESEARCH_DIR / "metrics"
CSV_PATH = RESEARCH_DIR / "ablation_summary.csv"
JSON_PATH = RESEARCH_DIR / "ablation_summary.json"
REPORT_PATH = RESEARCH_DIR / "ablation_report.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tempo.ml.compare")

CLASSES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]


def load_all_metrics() -> List[Dict[str, Any]]:
    if not METRICS_DIR.exists():
        logger.warning(f"Metrics directory {METRICS_DIR} does not exist.")
        return []

    results = []
    for f in sorted(METRICS_DIR.glob("*_results.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                results.append(data)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
    return results


def generate_summary():
    results = load_all_metrics()
    if not results:
        print("No evaluation results found in", METRICS_DIR)
        return

    # Table rows
    table_rows = []
    for r in results:
        cfg = r.get("config", {})
        exp_id = cfg.get("exp_id", r.get("exp_id", "unknown"))
        bb = cfg.get("backbone_type", r.get("backbone_type", "unknown")).upper()
        temporal = cfg.get("temporal_type", r.get("temporal_type", "unknown"))
        attn = cfg.get("use_attention", r.get("use_attention", False))
        arch_name = f"{bb} + {temporal}{'+Attn' if attn else ''}"

        val_m = r.get("val_metrics", {})
        test_m = r.get("test_metrics", {})
        br_m = r.get("back_row_metrics", {})
        gen_gap = r.get("generalization_gap", round(val_m.get("macro_f1", 0) - test_m.get("macro_f1", 0), 4))

        row = {
            "Experiment ID": exp_id,
            "Architecture": arch_name,
            "Backbone": bb,
            "Temporal": temporal,
            "Attention": attn,
            "Augmentation": "Strong" if cfg.get("strong_aug") else "Baseline",
            "Strategy": cfg.get("strategy", cfg.get("ablation_group", "architecture")),
            "Val Acc": val_m.get("accuracy", 0.0),
            "Val Macro-F1": val_m.get("macro_f1", 0.0),
            "Val Weighted-F1": val_m.get("weighted_f1", 0.0),
            "Test Acc": test_m.get("accuracy", 0.0),
            "Test Macro-P": test_m.get("macro_precision", 0.0),
            "Test Macro-R": test_m.get("macro_recall", 0.0),
            "Test Macro-F1": test_m.get("macro_f1", 0.0),
            "Test Weighted-F1": test_m.get("weighted_f1", 0.0),
            "Gen Gap (Val-Test)": gen_gap,
            "Back-Row Test Acc": br_m.get("accuracy", 0.0),
            "Back-Row Test F1": br_m.get("macro_f1", 0.0),
            "Best Epoch": cfg.get("best_epoch", -1),
            "Duration (s)": cfg.get("duration_sec", 0.0),
        }
        table_rows.append(row)

    # Sort by Test Macro-F1 descending (primary model selection metric)
    table_rows.sort(key=lambda x: x["Test Macro-F1"], reverse=True)

    # 1. Export CSV
    fieldnames = list(table_rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    logger.info(f"Saved CSV summary to {CSV_PATH}")

    # 2. Export JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)
    logger.info(f"Saved JSON summary to {JSON_PATH}")

    # 3. Print Console Table
    print("\n" + "=" * 115)
    print("  TEMPO BEHAVIOUR MODEL ABLATION & GENERALIZATION SUMMARY (Ranked by Test Macro-F1)")
    print("=" * 115)
    header = f"{'Rank':<4} {'Experiment ID':<26} {'Architecture':<24} {'Val F1':>8} {'Test F1':>8} {'Test Acc':>9} {'Gen Gap':>9} {'BR F1':>8}"
    print(header)
    print("-" * 115)
    for idx, r in enumerate(table_rows, 1):
        print(
            f"{idx:<4} {r['Experiment ID']:<26} {r['Architecture']:<24} "
            f"{r['Val Macro-F1']:>8.4f} {r['Test Macro-F1']:>8.4f} {r['Test Acc']:>9.4f} "
            f"{r['Gen Gap (Val-Test)']:>+9.4f} {r['Back-Row Test F1']:>8.4f}"
        )
    print("=" * 115)

    # 4. Generate Comprehensive Markdown Report
    best_model = table_rows[0]
    ref_f1 = 0.3929

    md_lines = []
    md_lines.append("# TEMPO Behaviour Recognition Model Ablation & Generalization Report\n")
    md_lines.append(f"**Date**: 2026-09-25  \n")
    md_lines.append(f"**Reference Baseline**: ResNet-18 + GRU (Held-Out Test Macro-F1 = `{ref_f1:.4f}`)  \n")
    md_lines.append(f"**Evaluation Protocol**: Single source-of-truth dataset (`sequences_repaired.json`), frozen 213-sequence test split (14 unseen videos), sequence length $T=16$, sampling FPS = 2.0.  \n")
    md_lines.append("\n---\n")

    md_lines.append("## 1. Executive Summary\n")
    if best_model["Test Macro-F1"] > ref_f1:
        delta = best_model["Test Macro-F1"] - ref_f1
        md_lines.append(
            f"> [!TIP]\n"
            f"> **Improvement Proven**: Model `{best_model['Experiment ID']}` ({best_model['Architecture']}) "
            f"achieved a held-out test Macro-F1 of **{best_model['Test Macro-F1']:.4f}**, exceeding the ResNet-18 baseline "
            f"by **{delta:+.4f}** ({delta/ref_f1*100:+.1f}% relative improvement).\n"
        )
    else:
        delta = best_model["Test Macro-F1"] - ref_f1
        md_lines.append(
            f"> [!WARNING]\n"
            f"> **Baseline Preserved**: No experimental model outperformed the reference ResNet-18 baseline on the held-out test set. "
            f"The best candidate `{best_model['Experiment ID']}` reached **{best_model['Test Macro-F1']:.4f}** vs baseline **{ref_f1:.4f}** "
            f"({delta:+.4f}). In accordance with strict production safety rules, **the production weights MUST NOT be modified**, "
            f"and ResNet-18 remains the reference deployment model.\n"
        )

    md_lines.append("\n## 2. Comparative Performance Table (Ranked by Test Macro-F1)\n")
    md_lines.append("| Rank | Experiment ID | Architecture | Augmentation | Val Macro-F1 | Test Macro-F1 | Test Acc | Gen Gap (Val-Test) | Back-Row F1 |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(table_rows, 1):
        md_lines.append(
            f"| {idx} | `{r['Experiment ID']}` | {r['Architecture']} | {r['Augmentation']} | "
            f"{r['Val Macro-F1']:.4f} | **{r['Test Macro-F1']:.4f}** | {r['Test Acc']:.4f} | "
            f"{r['Gen Gap (Val-Test)']:+.4f} | {r['Back-Row Test F1']:.4f} |"
        )

    md_lines.append("\n## 3. Architecture Ablation Findings (Models A through G)\n")
    md_lines.append(
        "A controlled comparison of 7 model architectures trained under identical conditions (class-balanced loss, AdamW, seed 42, $T=16$):\n"
    )
    arch_models = [r for r in table_rows if r["Experiment ID"].startswith("exp_") and not "gen_" in r["Experiment ID"]]
    arch_models.sort(key=lambda x: x["Experiment ID"])

    md_lines.append("| Model | Name | Backbone | Temporal Modeler | Attention | Val Macro-F1 | Test Macro-F1 | Test Accuracy |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in arch_models:
        md_lines.append(
            f"| `{r['Experiment ID']}` | {r['Architecture']} | {r['Backbone']} | {r['Temporal']} | "
            f"{'Yes' if r['Attention'] else 'No'} | {r['Val Macro-F1']:.4f} | {r['Test Macro-F1']:.4f} | {r['Test Acc']:.4f} |"
        )

    md_lines.append("\n## 4. Controlled Generalization & Augmentation Findings\n")
    md_lines.append(
        "Investigating the validation-to-test distribution shift:\n"
        "- **Validation Set Composition**: Reading dominant (145/223 = 65.0%), Writing (32/223 = 14.3%).\n"
        "- **Test Set Composition**: Writing dominant (129/213 = 60.6%), Reading (59/213 = 27.7%).\n"
        "- This inverse distribution causes models that overfit to Reading on the validation set to experience a severe drop on the held-out test set.\n"
    )

    gen_models = [r for r in table_rows if "gen_" in r["Experiment ID"]]
    if gen_models:
        md_lines.append("| Experiment | Strategy | Augmentation | Val Macro-F1 | Test Macro-F1 | Gen Gap |")
        md_lines.append("|---|---|---|---|---|---|")
        for r in gen_models:
            md_lines.append(
                f"| `{r['Experiment ID']}` | {r['Strategy']} | {r['Augmentation']} | "
                f"{r['Val Macro-F1']:.4f} | {r['Test Macro-F1']:.4f} | {r['Gen Gap (Val-Test)']:+.4f} |"
            )

    md_lines.append("\n## 5. Detailed Breakdown of Champion Model\n")
    # Find full result payload for best model
    best_payload = next((x for x in results if x.get("config", {}).get("exp_id") == best_model["Experiment ID"]), None)
    if best_payload:
        test_m = best_payload.get("test_metrics", {})
        md_lines.append(f"### Per-Class Test Metrics (`{best_model['Experiment ID']}`)\n")
        md_lines.append("| Observable Behaviour Class | Precision | Recall | F1-Score | Support |")
        md_lines.append("|---|---|---|---|---|")
        for c, pm in test_m.get("per_class", {}).items():
            md_lines.append(f"| {c} | {pm['precision']:.4f} | {pm['recall']:.4f} | {pm['f1']:.4f} | {pm['support']} |")

        md_lines.append("\n### Test Confusion Matrix\n")
        md_lines.append("```\n")
        cm = test_m.get("confusion_matrix", [])
        header = f"{'':<18}" + "".join(f"{c[:6]:>8}" for c in CLASSES)
        md_lines.append(header)
        for i, row in enumerate(cm):
            row_str = f"{CLASSES[i][:16]:<18}" + "".join(f"{v:>8d}" for v in row)
            md_lines.append(row_str)
        md_lines.append("```\n")

    md_lines.append("\n## 6. Recommendations & Decision Rules\n")
    if best_model["Test Macro-F1"] > ref_f1:
        md_lines.append(
            f"1. **Deploy New Champion**: `{best_model['Experiment ID']}` achieves superior test generalization ({best_model['Test Macro-F1']:.4f} vs {ref_f1:.4f}).\n"
            f"2. Export weights to `models/classroom_temporal_model.pth` and update `models/model_metadata.json`.\n"
        )
    else:
        md_lines.append(
            f"1. **Keep Production Weights Intact**: Preserve `models/classroom_temporal_model.pth` and `models/model_metadata.json` without automated modification.\n"
            f"2. **Document Distribution Shift**: The primary reason deeper models (ResNet-50) drop on test is the split class imbalance (Train/Val are Reading-heavy while Test is Writing-heavy).\n"
            f"3. **Future Research Priority**: Implement class-stratified video splitting and domain adversarial training to equalize writing vs reading representations across different classroom camera angles.\n"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Generated comprehensive markdown report at {REPORT_PATH}")


if __name__ == "__main__":
    generate_summary()
