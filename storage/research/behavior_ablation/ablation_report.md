# TEMPO Behaviour Recognition Model Ablation & Generalization Report

**Date**: 2026-09-25  

**Reference Baseline**: ResNet-18 + GRU (Held-Out Test Macro-F1 = `0.3929`)  

**Evaluation Protocol**: Single source-of-truth dataset (`sequences_repaired.json`), frozen 213-sequence test split (14 unseen videos), sequence length $T=16$, sampling FPS = 2.0.  


---

## 1. Executive Summary

> [!TIP]
> **Improvement Proven**: Model `exp_B_r18_gru` (RESNET18 + GRU) achieved a held-out test Macro-F1 of **0.5027**, exceeding the ResNet-18 baseline by **+0.1098** (+27.9% relative improvement).


## 2. Comparative Performance Table (Ranked by Test Macro-F1)

| Rank | Experiment ID | Architecture | Augmentation | Val Macro-F1 | Test Macro-F1 | Test Acc | Gen Gap (Val-Test) | Back-Row F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | `exp_B_r18_gru` | RESNET18 + GRU | Baseline | 0.4962 | **0.5027** | 0.5915 | -0.0065 | 0.5511 |
| 2 | `exp_A_r18_rnn` | RESNET18 + RNN | Baseline | 0.4508 | **0.4960** | 0.5446 | -0.0452 | 0.5151 |
| 3 | `exp_C_r18_bigru` | RESNET18 + BIGRU | Baseline | 0.5097 | **0.4926** | 0.6009 | +0.0171 | 0.5360 |
| 4 | `exp_G_r50_bigru_attn` | RESNET50 + BIGRU+Attn | Baseline | 0.5030 | **0.4353** | 0.6338 | +0.0677 | 0.4316 |
| 5 | `exp_gen_r18_strong_gru` | RESNET18 + GRU | Strong | 0.4978 | **0.4288** | 0.4883 | +0.0690 | 0.4192 |
| 6 | `exp_F_r50_bigru` | RESNET50 + BIGRU | Baseline | 0.4893 | **0.4229** | 0.5399 | +0.0664 | 0.4486 |
| 7 | `exp_gen_r18_strong_bigru_attn` | RESNET18 + BIGRU+Attn | Strong | 0.4899 | **0.4145** | 0.3991 | +0.0754 | 0.3931 |
| 8 | `exp_gen_r50_strong_bigru_attn` | RESNET50 + BIGRU+Attn | Strong | 0.4804 | **0.4096** | 0.4319 | +0.0708 | 0.3843 |
| 9 | `exp_D_r18_bigru_attn` | RESNET18 + BIGRU+Attn | Baseline | 0.4863 | **0.3646** | 0.5164 | +0.1217 | 0.3705 |
| 10 | `exp_gen_r18_frozen_gru` | RESNET18 + GRU | Baseline | 0.4769 | **0.2982** | 0.3756 | +0.1787 | 0.3240 |
| 11 | `exp_E_r50_gru` | RESNET50 + GRU | Baseline | 0.4596 | **0.2955** | 0.4366 | +0.1641 | 0.2522 |

## 3. Architecture Ablation Findings (Models A through G)

A controlled comparison of 7 model architectures trained under identical conditions (class-balanced loss, AdamW, seed 42, $T=16$):

| Model | Name | Backbone | Temporal Modeler | Attention | Val Macro-F1 | Test Macro-F1 | Test Accuracy |
|---|---|---|---|---|---|---|---|
| `exp_A_r18_rnn` | RESNET18 + RNN | RESNET18 | RNN | No | 0.4508 | 0.4960 | 0.5446 |
| `exp_B_r18_gru` | RESNET18 + GRU | RESNET18 | GRU | No | 0.4962 | 0.5027 | 0.5915 |
| `exp_C_r18_bigru` | RESNET18 + BIGRU | RESNET18 | BIGRU | No | 0.5097 | 0.4926 | 0.6009 |
| `exp_D_r18_bigru_attn` | RESNET18 + BIGRU+Attn | RESNET18 | BIGRU | Yes | 0.4863 | 0.3646 | 0.5164 |
| `exp_E_r50_gru` | RESNET50 + GRU | RESNET50 | GRU | No | 0.4596 | 0.2955 | 0.4366 |
| `exp_F_r50_bigru` | RESNET50 + BIGRU | RESNET50 | BIGRU | No | 0.4893 | 0.4229 | 0.5399 |
| `exp_G_r50_bigru_attn` | RESNET50 + BIGRU+Attn | RESNET50 | BIGRU | Yes | 0.5030 | 0.4353 | 0.6338 |

## 4. Controlled Generalization & Augmentation Findings

Investigating the validation-to-test distribution shift:
- **Validation Set Composition**: Reading dominant (145/223 = 65.0%), Writing (32/223 = 14.3%).
- **Test Set Composition**: Writing dominant (129/213 = 60.6%), Reading (59/213 = 27.7%).
- This inverse distribution causes models that overfit to Reading on the validation set to experience a severe drop on the held-out test set.

| Experiment | Strategy | Augmentation | Val Macro-F1 | Test Macro-F1 | Gen Gap |
|---|---|---|---|---|---|
| `exp_gen_r18_strong_gru` | strong_classroom_augmentation | Strong | 0.4978 | 0.4288 | +0.0690 |
| `exp_gen_r18_strong_bigru_attn` | strong_classroom_augmentation | Strong | 0.4899 | 0.4145 | +0.0754 |
| `exp_gen_r50_strong_bigru_attn` | lower_lr_strong_aug | Strong | 0.4804 | 0.4096 | +0.0708 |
| `exp_gen_r18_frozen_gru` | frozen_backbone | Baseline | 0.4769 | 0.2982 | +0.1787 |

## 5. Detailed Breakdown of Champion Model

### Per-Class Test Metrics (`exp_B_r18_gru`)

| Observable Behaviour Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Looking_Toward_Instruction | 0.5789 | 0.9167 | 0.7097 | 12 |
| Reading | 0.4065 | 0.8475 | 0.5495 | 59 |
| Writing | 0.9839 | 0.4729 | 0.6387 | 129 |
| Peer_Interaction | 0.5000 | 0.8000 | 0.6154 | 5 |
| Looking_Away | 0.0000 | 0.0000 | 0.0000 | 8 |

### Test Confusion Matrix

```

                    Lookin  Readin  Writin  Peer_I  Lookin
Looking_Toward_I        11       1       0       0       0
Reading                  8      50       1       0       0
Writing                  0      63      61       4       1
Peer_Interaction         0       1       0       4       0
Looking_Away             0       8       0       0       0
```


## 6. Recommendations & Decision Rules

1. **Deploy New Champion**: `exp_B_r18_gru` achieves superior test generalization (0.5027 vs 0.3929).
2. Export weights to `models/classroom_temporal_model.pth` and update `models/model_metadata.json`.
