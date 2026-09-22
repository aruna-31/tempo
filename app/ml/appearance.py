"""Short-term, non-biometric appearance descriptors for anonymous tracking.

The descriptor is computed from the full person crop, never from a face crop,
and is kept in memory only for the current tracking session. It is not stored
in PostgreSQL and is not suitable for cross-day identity.
"""

from typing import List

import cv2
import numpy as np


class AnonymousAppearanceEmbedder:
    def __init__(self, image_size: int = 16):
        self.image_size = image_size

    def __call__(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width - 1, x1)); y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2)); y2 = max(y1 + 1, min(height, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(96, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        resized = cv2.resize(hsv, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        pixels = resized.astype(np.float32).reshape(-1, 3)
        descriptor = np.concatenate([
            pixels.mean(axis=0) / np.array([180.0, 255.0, 255.0], dtype=np.float32),
            pixels.std(axis=0) / np.array([180.0, 255.0, 255.0], dtype=np.float32),
            resized.reshape(-1).astype(np.float32) / np.array([180.0, 255.0, 255.0] * (self.image_size * self.image_size), dtype=np.float32),
        ])
        norm = np.linalg.norm(descriptor)
        return descriptor / norm if norm > 0 else descriptor


def appearance_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first is None or second is None or first.size == 0 or second.size == 0:
        return 1.0
    return float(1.0 - np.clip(np.dot(first, second), -1.0, 1.0))