import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from app.ml.model import CLASS_NAMES, NUM_CLASSES
from app.ml.preprocessor import preprocessor

logger = logging.getLogger("tempo.ml.dataset")


class StudentTemporalDataset(Dataset):
    """
    PyTorch Dataset for individual student temporal behaviour sequences.
    Yields:
        sequence_tensor: (seq_len, 3, 224, 224)
        label_idx: int (0 to 4)
        metadata: Dict with session_id, student_track_id, timestamp, etc.
    """

    def __init__(self, sequences: List[Dict[str, Any]], augment: bool = False):
        self.sequences = sequences
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        item = self.sequences[idx]
        crops = item["crops"]  # List of numpy arrays (H, W, 3)
        label_name = item["label"]
        label_idx = CLASS_NAMES.index(label_name)

        # Preprocess frames into normalized PyTorch tensors: (seq_len, 3, 224, 224)
        seq_tensor = preprocessor.preprocess_batch(crops)

        if self.augment:
            # Subtle photometric noise augmentation for training robustness
            noise = torch.randn_like(seq_tensor) * 0.02
            seq_tensor = torch.clamp(seq_tensor + noise, -3.0, 3.0)

        metadata = {
            "session_id": item.get("session_id", 0),
            "track_id": item.get("track_id", 1),
            "label_name": label_name,
            "timestamp": item.get("timestamp", 0.0)
        }

        return seq_tensor, label_idx, metadata


class SyntheticClassroomSequenceGenerator:
    """
    Generates realistic student-track temporal sequences across multiple distinct classroom sessions.
    Strictly splits by session/recording ID to prevent cross-frame data leakage.
    """

    def __init__(self, sequence_length: int = 16, samples_per_session: int = 40):
        self.sequence_length = sequence_length
        self.samples_per_session = samples_per_session

    def generate_synthetic_dataset(
        self,
        num_sessions: int = 15,
        train_ratio: float = 0.65,
        val_ratio: float = 0.20
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Creates diverse student track temporal sequences grouped into distinct sessions.
        Returns:
            train_seqs, val_seqs, test_seqs (Zero overlap between sessions)
        """
        all_sessions: Dict[int, List[Dict[str, Any]]] = {}

        for sess_id in range(1, num_sessions + 1):
            session_samples = []
            # Each session has 3-5 distinct student tracks
            num_students = 4

            for track_id in range(1, num_students + 1):
                # For each student track in the session, generate multiple temporal samples for each behaviour
                for class_idx, class_name in enumerate(CLASS_NAMES):
                    samples_per_class = max(2, self.samples_per_session // (num_students * len(CLASS_NAMES)))
                    for s in range(samples_per_class):
                        t_start = (s * 4.0) + (class_idx * 15.0)

                        # Generate sequence of sequence_length frames simulating characteristic behavior patterns
                        crops = self._generate_behavior_crops(class_idx, self.sequence_length, sess_id, track_id)

                        sample = {
                            "session_id": sess_id,
                            "track_id": track_id,
                            "label": class_name,
                            "timestamp": round(t_start, 2),
                            "crops": crops
                        }
                        session_samples.append(sample)

            all_sessions[sess_id] = session_samples

        # Enforce recording/session-level split (Zero Data Leakage Protocol)
        session_ids = list(all_sessions.keys())
        np.random.seed(42)
        np.random.shuffle(session_ids)

        n_train = int(len(session_ids) * train_ratio)
        n_val = int(len(session_ids) * val_ratio)

        train_sess = session_ids[:n_train]
        val_sess = session_ids[n_train:n_train + n_val]
        test_sess = session_ids[n_train + n_val:]

        train_seqs = [s for sess in train_sess for s in all_sessions[sess]]
        val_seqs = [s for sess in val_sess for s in all_sessions[sess]]
        test_seqs = [s for sess in test_sess for s in all_sessions[sess]]

        logger.info(
            f"Dataset Partitioning (Session-Level Split): Train={len(train_seqs)} ({len(train_sess)} sessions), "
            f"Val={len(val_seqs)} ({len(val_sess)} sessions), Test={len(test_seqs)} ({len(test_sess)} sessions)"
        )

        return train_seqs, val_seqs, test_seqs

    def _generate_behavior_crops(
        self,
        class_idx: int,
        seq_len: int,
        sess_id: int,
        track_id: int
    ) -> List[np.ndarray]:
        """
        Synthesizes visual patterns corresponding to each behavior class:
        0: Looking_Toward_Instruction - upright posture, forward gaze, steady center
        1: Reading - lowered gaze, sustained orientation, static head
        2: Writing - downward tilt, slight rhythmic hand/wrist motion
        3: Peer_Interaction - lateral rotation, horizontal head turns
        4: Looking_Away - off-axis gaze, looking upward or away from study area
        """
        crops = []
        base_color = (
            (sess_id * 37 + track_id * 53) % 200 + 40,
            (sess_id * 61 + track_id * 29) % 200 + 40,
            (sess_id * 17 + track_id * 71) % 200 + 40
        )

        for t in range(seq_len):
            crop = np.full((224, 224, 3), fill_value=base_color, dtype=np.uint8)

            # Draw synthetic head/desk region
            cv2_center_x = 112
            cv2_center_y = 112

            if class_idx == 0:  # Looking Toward Instruction (centered, steady)
                cv2_center_y += int(np.sin(t * 0.1) * 2)
            elif class_idx == 1:  # Reading (tilted down, steady)
                cv2_center_y += 30
            elif class_idx == 2:  # Writing (tilted down with rhythmic hand motion)
                cv2_center_y += 35
                cv2_center_x += int(np.sin(t * 0.8) * 8)
            elif class_idx == 3:  # Peer Interaction (side orientation)
                cv2_center_x += 45 + int(np.sin(t * 0.3) * 5)
            elif class_idx == 4:  # Looking Away (turned far away or up)
                cv2_center_y -= 40
                cv2_center_x -= 30

            # Add simple spatial geometry
            x_min = max(10, cv2_center_x - 40)
            x_max = min(214, cv2_center_x + 40)
            y_min = max(10, cv2_center_y - 40)
            y_max = min(214, cv2_center_y + 40)
            crop[y_min:y_max, x_min:x_max] = (
                (base_color[0] + 50) % 255,
                (base_color[1] + 80) % 255,
                (base_color[2] + 40) % 255
            )

            crops.append(crop)

        return crops
