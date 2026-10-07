import logging
import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

logger = logging.getLogger("tempo.ml.model")

NUM_CLASSES = 5
CLASS_NAMES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away"
]


class ResNet18TemporalModel(nn.Module):
    """
    Classroom Temporal Behaviour Model.
    Architecture:
      1. Spatial Feature Extractor: ResNet-18 (512-dim feature embedding per frame)
      2. Temporal Sequence Modeler: GRU / LSTM / RNN (sequence length T -> hidden_dim)
      3. Classification Head: Linear -> 5 Observable Behaviour Classes
    """

    def __init__(
        self,
        temporal_type: str = "GRU",
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        use_pretrained_backbone: bool = False
    ):
        super().__init__()
        self.temporal_type = temporal_type.upper()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 1. Spatial Backbone (ResNet-18)
        if use_pretrained_backbone:
            try:
                base_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                base_resnet = resnet18(weights=None)
        else:
            base_resnet = resnet18(weights=None)

        # Retain feature extractor up to avgpool (512 dimensions)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        self.feature_dim = 512

        # 2. Temporal Sequence Module (GRU / LSTM / RNN)
        if self.temporal_type == "LSTM":
            self.temporal = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.temporal_type == "RNN":
            self.temporal = nn.RNN(
                input_size=self.feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="relu"
            )
        else:  # Default: GRU
            self.temporal = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract 512-dim features for batch of frames.
        Args:
            frames: (B, 3, 224, 224)
        Returns:
            features: (B, 512)
        """
        raw_feats = self.backbone(frames)  # (B, 512, 1, 1)
        return torch.flatten(raw_feats, 1)  # (B, 512)

    def forward_features(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from pre-extracted sequence features.
        Args:
            sequence_features: (B, seq_len, 512)
        Returns:
            logits: (B, num_classes)
        """
        if self.temporal_type == "LSTM":
            temporal_out, (hn, cn) = self.temporal(sequence_features)
        else:
            temporal_out, hn = self.temporal(sequence_features)

        # Use last timestep representation: (B, hidden_dim)
        last_step = temporal_out[:, -1, :]
        logits = self.classifier(last_step)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full end-to-end forward pass.
        Args:
            x: (B, seq_len, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
        """
        b, seq_len, c, h, w = x.shape
        # Reshape to (B * seq_len, 3, 224, 224) for spatial feature extraction
        flat_frames = x.view(b * seq_len, c, h, w)
        spatial_feats = self.extract_frame_features(flat_frames)  # (B * seq_len, 512)
        # Reshape back to sequence: (B, seq_len, 512)
        seq_feats = spatial_feats.view(b, seq_len, self.feature_dim)
        return self.forward_features(seq_feats)

    def load_trained_weights(self, weights_path: str) -> bool:
        """
        Strictly loads trained weights file.
        Raises FileNotFoundError if the file does not exist.
        Raises RuntimeError if the checkpoint is invalid or fails strict state_dict matching.
        """
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model checkpoint file not found at '{weights_path}'")

        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Strict state dict loading - mismatch will raise RuntimeError
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Successfully loaded model weights strictly from '{weights_path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint from '{weights_path}': {e}")
            raise RuntimeError(
                f"Model weights loading failed for '{weights_path}'. Architecture mismatch or corrupted weights: {e}"
            ) from e

    @classmethod
    def from_metadata(cls, metadata: Dict, weights_path: Optional[str] = None) -> "ResNet18TemporalModel":
        """
        Factory method to instantiate model strictly based on model_metadata.json.
        """
        temporal_type = metadata.get("temporal_model_type", "RNN")
        hidden_dim = metadata.get("hidden_dim", 128)
        num_layers = metadata.get("num_layers", 2)
        classes = metadata.get("classes", CLASS_NAMES)

        model = cls(
            temporal_type=temporal_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=len(classes)
        )
        if weights_path:
            model.load_trained_weights(weights_path)
        return model


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism over student sequence representations.
    Computes normalized attention coefficients across sequence timesteps
    and generates an aggregated context vector.
    """

    def __init__(self, feature_dim: int = 512, attention_dim: int = 128):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, feature_dim) sequence representations
        Returns:
            context: (B, feature_dim) attention-weighted summary vector
            weights: (B, T) normalized attention distribution
        """
        scores = self.attn_net(x)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(x * weights, dim=1)  # (B, feature_dim)
        return context, weights.squeeze(-1)


class ResNet50BiGRUTemporalModel(nn.Module):
    """
    Upgraded Classroom Temporal Behaviour Recognition Model.
    Architecture:
      1. Anonymous Student Crop -> ResNet-50 spatial backbone (2048-dim embedding)
      2. Feature Projection -> Linear(2048, 512) + LayerNorm(512) + GELU + Dropout
      3. Temporal Sequence Modeler -> 2-layer Bidirectional GRU (hidden=256 -> 512 output)
      4. Temporal Attention -> Self-attention weighting over sequence length T
      5. Classification Head -> Linear(512, 128) + LayerNorm + ReLU + Linear(128, 5)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        use_pretrained_backbone: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.feature_dim = 2048  # ResNet-50 feature dimension
        self.proj_dim = 512

        # 1. Spatial Backbone (ResNet-50)
        if use_pretrained_backbone:
            try:
                base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception:
                base_resnet = resnet50(weights=None)
        else:
            base_resnet = resnet50(weights=None)

        # Retain feature extractor up to avgpool (2048 dimensions)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])

        # 2. Feature Projection + LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3. 2-Layer Bidirectional GRU (hidden=256 -> bidirectional output dimension = 512)
        self.bigru = nn.GRU(
            input_size=self.proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.gru_out_dim = hidden_dim * 2  # 512

        # 4. Temporal Self-Attention
        self.attention = TemporalAttention(feature_dim=self.gru_out_dim, attention_dim=128)

        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dim features for batch of frames from ResNet-50.
        Args:
            frames: (B, 3, 224, 224)
        Returns:
            features: (B, 2048)
        """
        raw_feats = self.backbone(frames)  # (B, 2048, 1, 1)
        return torch.flatten(raw_feats, 1)  # (B, 2048)

    def forward_features(
        self,
        sequence_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass from pre-extracted sequence features.
        Args:
            sequence_features: (B, seq_len, 2048)
            return_attention: if True, returns (logits, attention_weights)
        Returns:
            logits: (B, num_classes) or (logits, attn_weights)
        """
        # Project: (B, T, 2048) -> (B, T, 512)
        proj = self.projection(sequence_features)

        # BiGRU: (B, T, 512) -> (B, T, 512)
        gru_out, _ = self.bigru(proj)

        # Temporal Attention: (B, T, 512) -> context: (B, 512), weights: (B, T)
        context, attn_weights = self.attention(gru_out)

        # Classifier: (B, 512) -> (B, num_classes)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full end-to-end forward pass.
        Args:
            x: (B, seq_len, 3, 224, 224)
            return_attention: whether to return temporal attention weights
        Returns:
            logits: (B, num_classes) or (logits, attn_weights)
        """
        b, seq_len, c, h, w = x.shape
        flat_frames = x.view(b * seq_len, c, h, w)
        spatial_feats = self.extract_frame_features(flat_frames)  # (B * seq_len, 2048)
        seq_feats = spatial_feats.view(b, seq_len, self.feature_dim)  # (B, seq_len, 2048)
        return self.forward_features(seq_feats, return_attention=return_attention)

    def load_trained_weights(self, weights_path: str) -> bool:
        """
        Strictly loads trained weights file.
        Raises FileNotFoundError if file is missing.
        Raises RuntimeError if weights mismatch or are corrupted.
        """
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model checkpoint file not found at '{weights_path}'")

        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Successfully loaded ResNet-50+BiGRU weights from '{weights_path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load ResNet-50 checkpoint from '{weights_path}': {e}")
            raise RuntimeError(
                f"Model weights loading failed for '{weights_path}'. Architecture mismatch or corrupted weights: {e}"
            ) from e

    @classmethod
    def from_metadata(cls, metadata: Dict, weights_path: Optional[str] = None) -> "ResNet50BiGRUTemporalModel":
        """
        Factory method to instantiate ResNet-50 model strictly based on model_metadata.json.
        """
        hidden_dim = metadata.get("hidden_dim", 256)
        num_layers = metadata.get("num_layers", 2)
        classes = metadata.get("classes", CLASS_NAMES)

        model = cls(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=len(classes),
            use_pretrained_backbone=False
        )
        if weights_path:
            model.load_trained_weights(weights_path)
        return model


def load_tempo_behaviour_model(metadata: Dict, weights_path: Optional[str] = None) -> nn.Module:
    """
    Unified factory method to instantiate the correct model architecture
    (ResNet-50 + BiGRU or ResNet-18) based on metadata specification.
    """
    backbone = metadata.get("spatial_backbone", "resnet18").lower()
    if backbone == "resnet50":
        return ResNet50BiGRUTemporalModel.from_metadata(metadata, weights_path=weights_path)
    return ResNet18TemporalModel.from_metadata(metadata, weights_path=weights_path)


