"""
Camera Source Abstraction for TEMPO Exam & Classroom Observation.

Supports unified frame ingestion across:
- MP4_DEMO: Local benchmark/demonstration video files.
- RTSP_STREAM: Direct IP camera streams.
- NVR_CHANNEL: Network Video Recorder channels.

Enables future deployment to live CCTV/NVR hardware without redesigning Edge or Fog pipelines.
"""
from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger("tempo.ml.camera")


class BaseCameraSource(ABC):
    """
    Abstract interface for camera video ingestion at Edge processing nodes.
    """

    def __init__(self, camera_id: str, device_code: str, name: str = "Camera"):
        self.camera_id = camera_id
        self.device_code = device_code
        self.name = name

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Returns the source type: MP4_DEMO, RTSP_STREAM, NVR_CHANNEL."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Checks if the video stream or demo file is accessible and decodable."""
        pass

    @abstractmethod
    def get_stream_info(self) -> Dict[str, Any]:
        """Returns metadata such as resolution, native FPS, source path/URL."""
        pass

    @abstractmethod
    def extract_frames(
        self,
        sampling_fps: float = 2.0,
        max_duration_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts sampled frames with frame indices, timestamps, and pixel data.
        Returns:
            List of dicts: [
                {"frame_number": int, "timestamp_seconds": float, "frame_data": np.ndarray}
            ]
        """
        pass

    @abstractmethod
    def get_source_reference(self) -> str:
        """Returns string reference (file path or RTSP URL) for audit/playback."""
        pass


class DemoMP4CameraSource(BaseCameraSource):
    """
    Demo camera source reading from a registered local MP4 file.
    Permits realistic edge evaluation, testing, and demonstrations.
    """

    def __init__(
        self,
        camera_id: str,
        device_code: str,
        video_path: str,
        name: str = "Demo Camera",
    ):
        super().__init__(camera_id, device_code, name)
        self.video_path = video_path

    @property
    def source_type(self) -> str:
        return "MP4_DEMO"

    def is_healthy(self) -> bool:
        if not os.path.exists(self.video_path):
            return False
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None

    def get_stream_info(self) -> Dict[str, Any]:
        if not os.path.exists(self.video_path):
            return {"healthy": False, "is_healthy": False, "error": f"File not found: {self.video_path}"}
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {"healthy": False, "is_healthy": False, "error": "Could not open video file"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(1.0, fps)
        cap.release()
        return {
            "healthy": True,
            "is_healthy": True,
            "source_type": self.source_type,
            "path": self.video_path,
            "width": width,
            "height": height,
            "fps": round(float(fps), 2),
            "duration_seconds": round(float(duration), 2),
            "total_frames": total_frames,
        }

    def extract_frames(
        self,
        sampling_fps: float = 2.0,
        max_duration_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Demo MP4 file '{self.video_path}' not found")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open demo video file '{self.video_path}'")

        frames_meta: List[Dict[str, Any]] = []
        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not native_fps or native_fps <= 0 or np.isnan(native_fps):
                native_fps = 30.0

            frame_interval = max(1, int(round(native_fps / max(0.1, sampling_fps))))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_interval == 0:
                    t_sec = float(frame_idx) / native_fps
                    if max_duration_seconds and t_sec > max_duration_seconds:
                        break
                    frames_meta.append({
                        "frame_number": frame_idx,
                        "timestamp_seconds": round(t_sec, 3),
                        "frame_data": frame,
                    })

                frame_idx += 1
        finally:
            cap.release()

        return frames_meta

    def get_source_reference(self) -> str:
        return self.video_path


class RTSPStreamCameraSource(BaseCameraSource):
    """
    RTSP stream source for live hardware IP cameras.
    """

    def __init__(
        self,
        camera_id: str,
        device_code: str,
        rtsp_url: str,
        name: str = "IP Camera",
    ):
        super().__init__(camera_id, device_code, name)
        self.rtsp_url = rtsp_url

    @property
    def source_type(self) -> str:
        return "RTSP_STREAM"

    def is_healthy(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        except Exception as e:
            logger.warning(f"RTSP check failed for {self.rtsp_url}: {e}")
            return False

    def get_stream_info(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            return {"healthy": False, "is_healthy": False, "error": f"Cannot connect to RTSP: {self.rtsp_url}"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "healthy": True,
            "is_healthy": True,
            "source_type": self.source_type,
            "rtsp_url": self.rtsp_url,
            "width": width,
            "height": height,
            "fps": round(float(fps), 2),
        }

    def extract_frames(
        self,
        sampling_fps: float = 2.0,
        max_duration_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise ConnectionError(f"Could not connect to RTSP stream '{self.rtsp_url}'")

        frames_meta: List[Dict[str, Any]] = []
        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = max(1, int(round(native_fps / max(0.1, sampling_fps))))
            frame_idx = 0

            max_frames = int(max_duration_seconds * native_fps) if max_duration_seconds else 600

            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx % frame_interval == 0:
                    t_sec = float(frame_idx) / native_fps
                    frames_meta.append({
                        "frame_number": frame_idx,
                        "timestamp_seconds": round(t_sec, 3),
                        "frame_data": frame,
                    })
                frame_idx += 1
        finally:
            cap.release()

        return frames_meta

    def get_source_reference(self) -> str:
        return self.rtsp_url


class CameraSourceFactory:
    """
    Factory to instantiate the appropriate BaseCameraSource from camera database record.
    Defaults to DemoMP4CameraSource if source is MP4_DEMO or unspecified.
    """

    DEFAULT_DEMO_PATH = "storage/test_videos/classroom_real_persons.mp4"

    @classmethod
    def create_from_camera(cls, camera: Any) -> BaseCameraSource:
        c_id = str(getattr(camera, "id", "default-cam"))
        code = getattr(camera, "device_code", "CAM-01")
        name = getattr(camera, "camera_name", "Registered Camera")
        source_type = (getattr(camera, "source_type", None) or "MP4_DEMO").upper()

        if source_type == "RTSP_STREAM":
            url = getattr(camera, "stream_url_or_channel", None) or ""
            return RTSPStreamCameraSource(camera_id=c_id, device_code=code, rtsp_url=url, name=name)
        else:
            # Default to DemoMP4CameraSource
            demo_path = getattr(camera, "demo_video_path", None) or cls.DEFAULT_DEMO_PATH
            if not os.path.exists(demo_path) and os.path.exists(cls.DEFAULT_DEMO_PATH):
                demo_path = cls.DEFAULT_DEMO_PATH
            return DemoMP4CameraSource(
                camera_id=c_id,
                device_code=code,
                video_path=demo_path,
                name=name,
            )
