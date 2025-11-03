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
fx_d, fy_d = 596.25827383, 593.35350108
cx_d, cy_d = 328.00224565, 246.72323964

dist_d = np.array([
    -5.75632759e-01,
     5.48853727e+00,
     3.71853753e-04,
     4.57792568e-03,
    -1.60971539e+01
])

# === RGB Camera Intrinsics ===
fx_rgb, fy_rgb = 501.3380852, 501.37135837
cx_rgb, cy_rgb = 326.20427782, 232.10061073

dist_rgb = np.array([
    -6.09913528e-02,
     6.20838340e-01,
    -9.85993593e-03,
     1.61211809e-03,
    -1.98825568e+00
])

# === Stereo Extrinsics (IR → RGB) ===
R = np.array([
    [ 0.99987482, -0.00253275,  0.01561793],
    [ 0.00218294,  0.99974728,  0.02237451],
    [-0.01567065, -0.02233761,  0.99962766]
])

T = np.array([
    [-0.02644189],
    [-0.00027204],
    [-0.01232775]
])

class KinectService:
    """Access to Kinect RGB/Depth frames and optional recording."""


    ############# SETUP #############


    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.videos_dir = os.path.join(self.base_dir, "videos")
        self.images_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Latest frames
        self._lock = threading.Lock()
        self._running = False
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_registered_depth: Optional[np.ndarray] = None
        self._latest_registered_rgb: Optional[np.ndarray] = None

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
            rgb, depth_raw = self._get_frames()
            if rgb is None or depth_raw is None:
                time.sleep(0.01)
                continue

            depth_m = self._depth_to_meters(depth_raw)

            # Only update the raw frames for streaming
            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth = depth_m

        if self._recording:
            self.stop_recording()

    ############# GETTING FRAMES #############

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

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._latest_rgb, self._latest_depth, self._latest_registered_rgb
    
    def get_latest_registered(self) -> Optional[np.ndarray]:
        """Get the latest RGB frame registered to depth coordinates."""
        with self._lock:
            return self._latest_registered_rgb
        

    ############# FILE ORGANIZATION #############


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
    

    ############# RECORDING/CAPTURE #############


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
            rgb = self._latest_rgb.copy()
            depth_m = self._latest_depth.copy()

        img_number = self._find_next_image_number()
        img_dir = os.path.join(self.images_dir, f"img_{img_number}")
        os.makedirs(img_dir, exist_ok=True)

        # Save RGB
        cv2.imwrite(os.path.join(img_dir, "rgb.png"), rgb)

        # Save depth (unregistered)
        depth_vis = cv2.convertScaleAbs(depth_m, alpha=0.03)
        cv2.imwrite(os.path.join(img_dir, "depth_raw_vis.png"), depth_vis)
        np.save(os.path.join(img_dir, "depth_raw_mm.npy"), depth_m)

        # Register RGB to depth
        try:
            registered_rgb = self._register_rgb_to_depth(rgb, depth_m)
            cv2.imwrite(os.path.join(img_dir, "rgb_registered.png"), registered_rgb)
        except Exception:
            registered_rgb = None

        return img_dir

        
    
    ############# REGISTRATION #############


    def _depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        depth_m = 1.0 / (depth_raw * -0.0030711016 + 3.3309495161)  # meters
        return depth_m.astype(np.float32)
        
    def _register_rgb_to_depth(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """
        Warp RGB into depth coordinates. Inputs:
        - rgb: H_rgb x W_rgb x 3 (uint8)
        - depth_m: H_depth x W_depth (float32), in meters
        Uses R,T which map from depth/IR -> RGB (i.e. point_in_rgb = R @ point_in_depth + T).
        """
        height, width = depth_m.shape
        # pixel grid in depth image
        j, i = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')  # j=rows, i=cols
        z = depth_m.flatten()  # length N in meters

        # back-project to depth/IR camera coords (points in meters)
        x = (i.flatten().astype(np.float32) - cx_d) * z / fx_d
        y = (j.flatten().astype(np.float32) - cy_d) * z / fy_d
        points_depth = np.stack((x, y, z), axis=1).T  # 3 x N

        # transform to RGB camera frame: IMPORTANT: use R @ p + T (not R.T)
        points_rgb = (R @ points_depth) + T  # 3 x N

        pts_rgb = points_rgb.T  # N x 3
        # Do pinhole projection (no distortion) then apply distortion if needed:
        x_rgb = pts_rgb[:, 0] / (pts_rgb[:, 2] + 1e-8)
        y_rgb = pts_rgb[:, 1] / (pts_rgb[:, 2] + 1e-8)
        u = (x_rgb * fx_rgb + cx_rgb)
        v = (y_rgb * fy_rgb + cy_rgb)

        # nearest-neighbor sampling into depth-grid sized rgb_registered image
        rgb_registered = np.zeros((height, width, 3), dtype=np.uint8)
        u_r = np.round(u).astype(np.int32)
        v_r = np.round(v).astype(np.int32)

        # valid mask where projected pixel falls inside rgb image bounds
        valid = (u_r >= 0) & (u_r < rgb.shape[1]) & (v_r >= 0) & (v_r < rgb.shape[0]) & (pts_rgb[:, 2] > 0)
        idxs = np.nonzero(valid)[0]

        # map flattened indices back to 2D depth coords
        y_img = idxs // width
        x_img = idxs % width
        rgb_registered[y_img, x_img] = rgb[v_r[idxs], u_r[idxs]]

        return rgb_registered
    
    def create_alignment_verification_image(self) -> Optional[np.ndarray]:
        """Return blended RGB (aligned to depth) and colorized depth for visual check."""
        if self._latest_registered_rgb is None or self._latest_depth is None:
            return None

        rgb_copy = self._latest_registered_rgb.copy()
        depth_vis = cv2.convertScaleAbs(self._latest_depth, alpha=0.03)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(rgb_copy, 0.5, depth_colored, 0.5, 0)
        return blended

    def verify_alignment_accuracy(self) -> dict:
        """Verify the accuracy of RGB/depth alignment using edge detection."""
        with self._lock:
            if self._latest_rgb is None or self._latest_registered_depth is None:
                return {"error": "No frames available"}
            
            # Convert to grayscale for edge detection
            rgb_gray = cv2.cvtColor(self._latest_rgb, cv2.COLOR_BGR2GRAY)
            depth_gray = cv2.convertScaleAbs(self._latest_registered_depth, alpha=0.03)
            
            # Apply edge detection
            rgb_edges = cv2.Canny(rgb_gray, 50, 150)
            depth_edges = cv2.Canny(depth_gray, 50, 150)
            
            # Calculate edge correlation
            correlation = cv2.matchTemplate(rgb_edges, depth_edges, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Count edge pixels
            rgb_edge_count = np.sum(rgb_edges > 0)
            depth_edge_count = np.sum(depth_edges > 0)
            
            # Calculate edge overlap
            edge_overlap = cv2.bitwise_and(rgb_edges, depth_edges)
            overlap_count = np.sum(edge_overlap > 0)
            
            return {
                "correlation": float(correlation),
                "rgb_edges": int(rgb_edge_count),
                "depth_edges": int(depth_edge_count),
                "overlap_edges": int(overlap_count),
                "overlap_percentage": float(overlap_count / max(rgb_edge_count, depth_edge_count, 1) * 100)
            }
        



