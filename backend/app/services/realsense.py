import threading
import time
import os
import csv
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except Exception:
    HAS_REALSENSE = False


class RealSenseService:
    """Thread-safe access to RealSense RGB/Depth frames and image capture."""

    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = base_dir
        self.rgb_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.csv_path = os.path.join(self.base_dir, "captures.csv")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['rgb_path', 'depth_path', 'height', 'sex'])

        self._lock = threading.Lock()
        self._running = False
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None

        # RealSense pipeline
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        
        if not HAS_REALSENSE:
            print("Warning: pyrealsense2 not available. RealSense service will not work.")
            return
        
        try:
            # Configure RealSense pipeline
            self._pipeline = rs.pipeline()
            config = rs.config()

            # Enable color and depth streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            # Start streaming
            profile = self._pipeline.start(config)

            # Create align object to register depth to color
            align_to = rs.stream.color
            self._align = rs.align(align_to)

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align = None

    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        while self._running:
            try:
                if not self._pipeline or not self._align:
                    time.sleep(0.01)
                    continue

                # Wait for frames and align them
                frames = self._pipeline.wait_for_frames()
                aligned_frames = self._align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                with self._lock:
                    self._latest_rgb = color_image
                    self._latest_depth = depth_image

            except Exception as e:
                print(f"Error in RealSense capture loop: {e}")
                time.sleep(0.1)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the latest RGB and depth frames."""
        with self._lock:
            return self._latest_rgb, self._latest_depth

    def capture_image(self, height: Optional[str] = None, sex: Optional[str] = None) -> Optional[dict]:
        """
        Capture current RGB image and depth array.
        Saves metadata to CSV file.
        Returns dict with paths to saved files, or None if capture failed.
        """
        with self._lock:
            if self._latest_rgb is None or self._latest_depth is None:
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save RGB image
            rgb_filename = f"{timestamp}_rgb.png"
            rgb_path = os.path.join(self.rgb_dir, rgb_filename)
            cv2.imwrite(rgb_path, self._latest_rgb)

            # Save depth array as .npy
            depth_filename = f"{timestamp}_depth.npy"
            depth_path = os.path.join(self.depth_dir, depth_filename)
            np.save(depth_path, self._latest_depth)

            # Append to CSV file
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    rgb_path,
                    depth_path,
                    height or '',
                    sex or ''
                ])

            return {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "timestamp": timestamp
            }
