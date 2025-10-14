import threading
import time
import os
from typing import Optional, Tuple

import numpy as np

try:
    import freenect  # type: ignore
    HAS_FREENECT = True
except Exception:
    HAS_FREENECT = False

import cv2


class KinectService:
    """Thread-safe access to Kinect RGB/Depth frames and optional recording."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.videos_dir = os.path.join(self.base_dir, "videos")
        self.images_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._running = False
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None

        # Recording state
        self._recording = False
        self._rgb_writer: Optional[cv2.VideoWriter] = None
        self._depth_writer: Optional[cv2.VideoWriter] = None
        self._record_depth_values: list[np.ndarray] = []
        self._current_take_dir: Optional[str] = None

        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._recording:
            self.stop_recording()

    def _capture_loop(self) -> None:
        while self._running:
            rgb, depth = self._get_frames()
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth = depth
                if self._recording and self._rgb_writer and self._depth_writer:
                    self._rgb_writer.write(rgb)
                    depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
                    self._depth_writer.write(depth_vis)
                    self._record_depth_values.append(depth.copy())
        # Ensure writers are closed if loop exits
        if self._recording:
            self.stop_recording()

    def _get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if HAS_FREENECT:
            rgb, _ = freenect.sync_get_video()
            depth, _ = freenect.sync_get_depth()
            if rgb is None or depth is None:
                return None, None
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return rgb_bgr, depth.astype(np.uint16)
        # Fallback to webcam for development when Kinect not available
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None
        # Create fake depth map for consistency
        fake_depth = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        return frame, fake_depth

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self._latest_rgb, self._latest_depth

    def _find_next_take_number(self) -> int:
        take_number = 1
        while os.path.exists(os.path.join(self.videos_dir, f"take_{take_number}")):
            take_number += 1
        return take_number

    def _find_next_image_number(self) -> int:
        img_number = 1
        while os.path.exists(os.path.join(self.images_dir, f"img_{img_number}")):
            img_number += 1
        return img_number

    def start_recording(self) -> Optional[str]:
        with self._lock:
            if self._recording:
                return self._current_take_dir
            take_number = self._find_next_take_number()
            take_dir = os.path.join(self.videos_dir, f"take_{take_number}")
            os.makedirs(take_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._rgb_writer = cv2.VideoWriter(os.path.join(take_dir, "rgb.avi"), fourcc, 30.0, (640, 480))
            self._depth_writer = cv2.VideoWriter(os.path.join(take_dir, "depth.avi"), fourcc, 30.0, (640, 480), False)
            self._record_depth_values = []
            self._current_take_dir = take_dir
            self._recording = True
            return take_dir

    def stop_recording(self) -> Optional[str]:
        with self._lock:
            if not self._recording:
                return None
            if self._rgb_writer:
                self._rgb_writer.release()
                self._rgb_writer = None
            if self._depth_writer:
                self._depth_writer.release()
                self._depth_writer = None
            # Save depth raw
            if self._current_take_dir is not None and self._record_depth_values:
                np.save(
                    os.path.join(self._current_take_dir, "depth_raw.npy"),
                    np.array(self._record_depth_values),
                )
            self._recording = False
            take_dir = self._current_take_dir
            self._current_take_dir = None
            self._record_depth_values = []
            return take_dir

    def capture_image(self) -> Optional[str]:
        with self._lock:
            if self._latest_rgb is None or self._latest_depth is None:
                return None
            img_number = self._find_next_image_number()
            img_dir = os.path.join(self.images_dir, f"img_{img_number}")
            os.makedirs(img_dir, exist_ok=True)
            cv2.imwrite(os.path.join(img_dir, "rgb.png"), self._latest_rgb)
            depth_vis = cv2.convertScaleAbs(self._latest_depth, alpha=0.03)
            cv2.imwrite(os.path.join(img_dir, "depth.png"), depth_vis)
            np.save(os.path.join(img_dir, "depth_raw.npy"), self._latest_depth)
            return img_dir


