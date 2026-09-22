import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from app.ml.model import CLASS_NAMES, NUM_CLASSES, ResNet18TemporalModel

logger = logging.getLogger("tempo.ml.train")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics:
    - Overall accuracy
    - Macro precision, recall, F1
    - Weighted precision, recall, F1
    - 5x5 Confusion Matrix
    - Per-class metrics
    - Confidence metrics
    """
    num_classes = len(CLASS_NAMES)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c_idx, c_name in enumerate(CLASS_NAMES):
        tp = conf_matrix[c_idx, c_idx]
        fp = np.sum(conf_matrix[:, c_idx]) - tp
        fn = np.sum(conf_matrix[c_idx, :]) - tp
        support = int(np.sum(conf_matrix[c_idx, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        per_class_metrics[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    total_samples = len(y_true)
    accuracy = float(np.sum(y_true == y_pred) / total_samples) if total_samples > 0 else 0.0
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_precision = float(np.sum(np.array(precisions) * np.array(supports)) / total_samples) if total_samples > 0 else 0.0
    weighted_recall = float(np.sum(np.array(recalls) * np.array(supports)) / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_samples) if total_samples > 0 else 0.0

    # Confidence calibration analysis
    confidences = np.max(y_probs, axis=1)
    correct_mask = (y_true == y_pred)
    avg_conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
    avg_conf_incorrect = float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": per_class_metrics,
        "confidence_analysis": {
            "avg_confidence_correct": round(avg_conf_correct, 4),
            "avg_confidence_incorrect": round(avg_conf_incorrect, 4),
            "overall_avg_confidence": round(float(np.mean(confidences)), 4)
        }
    }


def analyze_difficult_examples_and_transitions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Analyzes ambiguous samples and common temporal error patterns.
    """
    ambiguous_examples = []
    for idx, (t, p, probs) in enumerate(zip(y_true, y_pred, y_probs)):
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])
        is_error = bool(t != p)
        meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}

        if is_error or margin < 0.25:
            ambiguous_examples.append({
                "index": idx,
                "session_id": meta.get("session_id", 0),
                "track_id": meta.get("track_id", 1),
                "timestamp": meta.get("timestamp", 0.0),
                "true_label": CLASS_NAMES[t],
                "pred_label": CLASS_NAMES[p],
                "top_confidence": round(float(sorted_probs[0]), 4),
                "second_confidence": round(float(sorted_probs[1]), 4),
                "margin": round(margin, 4),
                "is_misclassified": is_error
            })

    ambiguous_examples.sort(key=lambda x: x["margin"])

    transition_errors: Dict[str, int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pair = f"{CLASS_NAMES[t]} -> {CLASS_NAMES[p]}"
            transition_errors[pair] = transition_errors.get(pair, 0) + 1

    return {
        "ambiguous_examples": ambiguous_examples[:top_k],
        "common_transition_errors": dict(sorted(transition_errors.items(), key=lambda x: x[1], reverse=True))
    }


def extract_features_for_dataset(
    model: ResNet18TemporalModel,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Extracts spatial 512-dim features for all sequences in a DataLoader using ResNet-18 backbone.
    Returns:
        features: (N, seq_len, 512)
        labels: (N,)
        metadatas: List[Dict]
    """
    model.eval()
    all_feats = []
    all_labels = []
    all_metas = []

    with torch.no_grad():
        for b_idx, (inputs, targets, metas) in enumerate(data_loader):
            b, seq_len, c, h, w = inputs.shape
            flat_inputs = inputs.view(b * seq_len, c, h, w).to(device)
            
            # Mini-chunk pass for optimal CPU execution
            chunk_size = 64
            chunk_list = []
            for start_i in range(0, flat_inputs.shape[0], chunk_size):
                sub_chunk = flat_inputs[start_i:start_i + chunk_size]
                chunk_list.append(model.extract_frame_features(sub_chunk))
            
            spatial_feats = torch.cat(chunk_list, dim=0)
            seq_feats = spatial_feats.view(b, seq_len, model.feature_dim).cpu()
            all_feats.append(seq_feats)
            all_labels.append(targets)

            batch_size = len(targets)
            for i in range(batch_size):
                all_metas.append({k: metas[k][i] for k in metas})
            
            if (b_idx + 1) % 2 == 0 or (b_idx + 1) == len(data_loader):
                print(f"      Extracted batch {b_idx + 1}/{len(data_loader)}", flush=True)

    feats_tensor = torch.cat(all_feats, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return feats_tensor, labels_tensor, all_metas


def evaluate_feature_model(
    model: ResNet18TemporalModel,
    feats: torch.Tensor,
    labels: torch.Tensor,
    metadatas: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates temporal model over pre-extracted feature tensors."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    n = len(labels)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_feats = feats[i:i + batch_size].to(device)
            batch_targets = labels[i:i + batch_size]

            logits = model.forward_features(batch_feats)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(batch_targets.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = compute_classification_metrics(y_true, y_pred, y_probs)
    error_analysis = analyze_difficult_examples_and_transitions(y_true, y_pred, y_probs, metadatas)

    return metrics, error_analysis


def train_single_model(
    temporal_type: str,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    val_metas: List[Dict[str, Any]],
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    test_metas: List[Dict[str, Any]],
    num_epochs: int = 25,
    lr: float = 3e-3,
    device: Optional[torch.device] = None,
    hidden_dim: int = 128
) -> Dict[str, Any]:
    """
    Trains and evaluates a specific ResNet18 + Temporal architecture on feature tensors.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18TemporalModel(
        temporal_type=temporal_type,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.25
    )
    model.to(dev)

    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    # Optimize temporal sequence module and classifier head
    trainable_params = list(model.temporal.parameters()) + list(model.classifier.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_time = time.time()
    best_val_f1 = -1.0
    best_state = None

    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)

            optimizer.zero_grad()
            logits = model.forward_features(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Validation evaluation
        val_metrics, _ = evaluate_feature_model(model, val_feats, val_labels, val_metas, dev)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = round(time.time() - start_time, 2)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_error_analysis = evaluate_feature_model(
        model, test_feats, test_labels, test_metas, dev
    )

    return {
        "architecture": f"ResNet18_{temporal_type}",
        "temporal_type": temporal_type,
        "train_duration_sec": train_duration,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_metrics": test_metrics,
        "error_analysis": test_error_analysis,
        "history": history,
        "model_state_dict": model.state_dict()
    }
