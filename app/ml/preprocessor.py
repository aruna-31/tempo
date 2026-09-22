import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Standard ImageNet transform for ResNet-18
DEFAULT_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_TENSOR = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FramePreprocessor:
    """Handles high-performance image/frame normalization and tensor batching."""

    def __init__(self, transform=None):
        self.transform = transform or DEFAULT_TRANSFORM

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a (3, 224, 224) normalized tensor."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def preprocess_numpy_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Converts OpenCV/NumPy BGR or RGB array to (3, 224, 224) normalized tensor."""
        if frame.shape[:2] == (224, 224) and frame.ndim == 3:
            # Fast vectorized path for standard size
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            return (t - MEAN_TENSOR) / STD_TENSOR

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = frame[:, :, ::-1]
            img = Image.fromarray(rgb_frame)
        else:
            img = Image.fromarray(frame)
        return self.preprocess_image(img)

    def preprocess_batch(self, frames: list) -> torch.Tensor:
        """
        Preprocesses a list of PIL Images or NumPy frames.
        Returns: (B, 3, 224, 224)
        """
        if not frames:
            return torch.empty((0, 3, 224, 224))

        # Check if all elements are (224, 224, 3) numpy arrays
        if isinstance(frames[0], np.ndarray) and frames[0].shape == (224, 224, 3):
            # Convert BGR -> RGB and permute to (B, 3, 224, 224)
            arr = np.stack(frames, axis=0)[:, :, :, ::-1].copy()
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            mean = MEAN_TENSOR.unsqueeze(0)  # (1, 3, 1, 1)
            std = STD_TENSOR.unsqueeze(0)
            return (t - mean) / std

        tensors = []
        for f in frames:
            if isinstance(f, Image.Image):
                tensors.append(self.preprocess_image(f))
            elif isinstance(f, np.ndarray):
                tensors.append(self.preprocess_numpy_frame(f))
            elif isinstance(f, torch.Tensor):
                tensors.append(f)
            else:
                raise ValueError(f"Unsupported frame type: {type(f)}")
        return torch.stack(tensors, dim=0)


preprocessor = FramePreprocessor()
