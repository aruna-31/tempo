import os
import pytest
import torch
from app.ml.model import (
    CLASS_NAMES,
    NUM_CLASSES,
    ResNet18TemporalModel,
    ResNet50BiGRUTemporalModel,
    TemporalAttention,
    load_tempo_behaviour_model,
)


def test_resnet50_bigru_attention_forward():
    """Verify end-to-end forward pass and attention weighting."""
    batch_size = 2
    seq_len = 16
    model = ResNet50BiGRUTemporalModel(
        hidden_dim=256,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_pretrained_backbone=False,
    )
    model.eval()

    dummy_input = torch.randn(batch_size, seq_len, 3, 224, 224)
    with torch.no_grad():
        logits, attn_weights = model(dummy_input, return_attention=True)

    assert logits.shape == (batch_size, 5)
    assert attn_weights.shape == (batch_size, seq_len)
    # Check that attention weights sum to 1.0 along the temporal dimension
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_resnet50_forward_features():
    """Verify forward_features from pre-extracted 2048-dim spatial embeddings."""
    batch_size = 4
    for seq_len in [8, 16, 24, 32]:
        model = ResNet50BiGRUTemporalModel(
            hidden_dim=256,
            num_layers=2,
            num_classes=NUM_CLASSES,
            use_pretrained_backbone=False,
        )
        model.eval()

        seq_feats = torch.randn(batch_size, seq_len, 2048)
        with torch.no_grad():
            logits, attn = model.forward_features(seq_feats, return_attention=True)

        assert logits.shape == (batch_size, 5)
        assert attn.shape == (batch_size, seq_len)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_temporal_attention_module():
    """Verify temporal attention module mechanism directly."""
    attn = TemporalAttention(feature_dim=512, attention_dim=128)
    x = torch.randn(3, 16, 512)
    context, weights = attn(x)

    assert context.shape == (3, 512)
    assert weights.shape == (3, 16)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_resnet50_gradient_flow():
    """Verify gradient flow through all components during backward pass."""
    model = ResNet50BiGRUTemporalModel(
        hidden_dim=256,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_pretrained_backbone=False,
    )
    model.train()

    seq_feats = torch.randn(2, 8, 2048, requires_grad=True)
    targets = torch.tensor([1, 3], dtype=torch.long)

    logits = model.forward_features(seq_feats)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    # Verify gradients exist and are non-zero in projection, bigru, attention, and classifier
    assert model.projection[0].weight.grad is not None
    assert torch.norm(model.projection[0].weight.grad) > 0

    assert model.bigru.weight_ih_l0.grad is not None
    assert torch.norm(model.bigru.weight_ih_l0.grad) > 0

    assert model.attention.attn_net[0].weight.grad is not None
    assert torch.norm(model.attention.attn_net[0].weight.grad) > 0

    assert model.classifier[-1].weight.grad is not None
    assert torch.norm(model.classifier[-1].weight.grad) > 0


def test_unified_model_factory():
    """Verify load_tempo_behaviour_model instantiates appropriate architecture."""
    # ResNet-50 metadata
    meta_50 = {
        "spatial_backbone": "resnet50",
        "temporal_model_type": "BiGRU",
        "hidden_dim": 256,
        "num_layers": 2,
        "classes": CLASS_NAMES,
    }
    m50 = load_tempo_behaviour_model(meta_50)
    assert isinstance(m50, ResNet50BiGRUTemporalModel)
    assert m50.hidden_dim == 256
    assert m50.num_layers == 2

    # ResNet-18 metadata
    meta_18 = {
        "spatial_backbone": "resnet18",
        "temporal_model_type": "GRU",
        "hidden_dim": 128,
        "num_layers": 2,
        "classes": CLASS_NAMES,
    }
    m18 = load_tempo_behaviour_model(meta_18)
    assert isinstance(m18, ResNet18TemporalModel)
    assert m18.hidden_dim == 128
