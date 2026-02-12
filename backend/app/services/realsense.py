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
        
        # Initialize CSV file with headers if it doesn't exist or has old headers
        expected_headers = ['rgb_path', 'depth_path', 'weight', 'race_ethnicity', 'activity_level', 'height', 'sex', 'processing_time_seconds', 'volume_cm3', 'estimated_weight_kg']
        needs_header_update = False
        
        if not os.path.exists(self.csv_path):
            needs_header_update = True
        else:
            # Check if file has correct headers
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    if header != expected_headers:
                        needs_header_update = True
            except (StopIteration, FileNotFoundError):
                # File is empty or doesn't exist
                needs_header_update = True
        
        if needs_header_update:
            # If file exists with old headers, migrate data
            if os.path.exists(self.csv_path):
                # Read existing rows
                existing_rows = []
                try:
                    with open(self.csv_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        old_header = next(reader)
                        # Read all existing data rows
                        for row in reader:
                            existing_rows.append(row)
                except StopIteration:
                    # File is empty
                    pass
                
                # Write new header and migrate existing rows
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(expected_headers)
                    # Migrate old rows: add empty values for new columns
                    for row in existing_rows:
                        # Old format: [rgb_path, depth_path, height, sex]
                        # New format: [rgb_path, depth_path, weight, race_ethnicity, activity_level, height, sex, processing_time_seconds, volume_cm3, estimated_weight_kg]
                        if len(row) == 4:
                            # Old format - insert empty values for new columns
                            writer.writerow([row[0], row[1], '', '', '', row[2], row[3], '', '', ''])
                        elif len(row) == 7:
                            # Format without processing columns - add empty processing columns
                            writer.writerow(row + ['', '', ''])
                        elif len(row) == len(expected_headers):
                            # Already in new format
                            writer.writerow(row)
                        else:
                            # Unknown format - try to preserve as much as possible
                            padded_row = row + [''] * (len(expected_headers) - len(row))
                            writer.writerow(padded_row[:len(expected_headers)])
            else:
                # Create new file with headers
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(expected_headers)

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
            # Check if any RealSense device is connected
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found. Please connect a RealSense camera.")
            
            print(f"Found {len(devices)} RealSense device(s)")
            device = devices[0]  # Use first available device
            
            # Query available stream profiles from device sensors
            color_profiles = []
            depth_profiles = []
            
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    if profile.is_video_stream_profile():
                        vp = profile.as_video_stream_profile()
                        if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                            color_profiles.append(vp)
                        elif profile.stream_type() == rs.stream.depth and profile.format() == rs.format.z16:
                            depth_profiles.append(vp)
            
            if not color_profiles:
                raise RuntimeError("No compatible color stream found on RealSense device.")
            if not depth_profiles:
                raise RuntimeError("No compatible depth stream found on RealSense device.")
            
            # Find a matching color and depth profile
            # Prefer 640x480 @ 30fps if available, otherwise use first available
            color_vp = None
            depth_vp = None
            
            for vp in color_profiles:
                if vp.width() == 1280 and vp.height() == 720 and vp.fps() == 6:
                    color_vp = vp
                    break
            if not color_vp:
                color_vp = color_profiles[0]
            
            for vp in depth_profiles:
                if vp.width() == 1280 and vp.height() == 720 and vp.fps() == 6:
                    depth_vp = vp
                    break
            if not depth_vp:
                depth_vp = depth_profiles[0]
            
            # Configure RealSense pipeline
            self._pipeline = rs.pipeline()
            config = rs.config()
            
            config.enable_stream(
                rs.stream.color,
                color_vp.width(),
                color_vp.height(),
                color_vp.format(),
                color_vp.fps()
            )
            config.enable_stream(
                rs.stream.depth,
                depth_vp.width(),
                depth_vp.height(),
                depth_vp.format(),
                depth_vp.fps()
            )
            
            print(f"Configuring RealSense streams:")
            print(f"  Color: {color_vp.width()}x{color_vp.height()} @ {color_vp.fps()}fps")
            print(f"  Depth: {depth_vp.width()}x{depth_vp.height()} @ {depth_vp.fps()}fps")

            # Try to start streaming with the configured streams
            try:
                profile = self._pipeline.start(config)
                print("RealSense pipeline started successfully")
            except RuntimeError as e:
                # If specific config fails, try with default/auto configuration
                error_msg = str(e).lower()
                if "no device" not in error_msg and "not found" not in error_msg:
                    print(f"Failed to start with specific config: {e}")
                    print("Trying with default configuration...")
                    # Try default configuration (let RealSense auto-detect)
                    config = rs.config()
                    config.enable_all_streams()
                    profile = self._pipeline.start(config)
                    print("RealSense pipeline started with default configuration")
                else:
                    raise

            # Create align object to register depth to color
            align_to = rs.stream.color

            # Print intrinsics
            col_stream = profile.get_stream(align_to)
            intr = col_stream.as_video_stream_profile().get_intrinsics()
            fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
            print(f"Camera intrinsics - fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")

            self._align = rs.align(align_to)
            
            # Wait for a few frames to stabilize (warm-up period)
            print("Waiting for frames to stabilize...")
            frames_received = 0
            for i in range(60):  # Wait up to 2 seconds
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=500)
                    if frames:
                        frames_received += 1
                        if frames_received >= 5:  # Got a few frames, good enough
                            print(f"Received {frames_received} frames, ready to stream")
                            break
                except RuntimeError:
                    # Timeout or other error, continue waiting
                    pass
                time.sleep(0.033)  # ~30fps check rate
            
            if frames_received == 0:
                print("Warning: No frames received during warm-up. Camera may not be ready.")
            else:
                print(f"Warm-up complete. Received {frames_received} frames.")
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            print("RealSense capture thread started")
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")
            import traceback
            traceback.print_exc()
            self._running = False
            if self._pipeline:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
            self._align = None
            raise

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
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self._running:
            try:
                if not self._pipeline or not self._align:
                    time.sleep(0.01)
                    continue

                # Wait for frames with timeout
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Warning: Too many consecutive frame timeouts. Check camera connection.")
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                aligned_frames = self._align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Warning: Missing depth or color frames. Check camera alignment.")
                        consecutive_errors = 0
                    continue

                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                with self._lock:
                    self._latest_rgb = color_image
                    self._latest_depth = depth_image
                
                # Reset error counter on successful frame capture
                consecutive_errors = 0

            except RuntimeError as e:
                # RuntimeError often indicates device disconnection
                error_msg = str(e)
                if "No frames" in error_msg or "timeout" in error_msg.lower():
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Error in RealSense capture loop: {e}")
                        print("Camera may be disconnected or not responding.")
                        consecutive_errors = 0
                    time.sleep(0.1)
                else:
                    print(f"RuntimeError in RealSense capture loop: {e}")
                    time.sleep(0.1)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Unexpected error in RealSense capture loop: {e}")
                    import traceback
                    traceback.print_exc()
                    consecutive_errors = 0
                time.sleep(0.1)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the latest RGB and depth frames."""
        with self._lock:
            return self._latest_rgb, self._latest_depth
    
    def is_running(self) -> bool:
        """Check if the RealSense service is running."""
        return self._running
    
    def has_frames(self) -> bool:
        """Check if frames are available."""
        with self._lock:
            return self._latest_rgb is not None and self._latest_depth is not None

    def capture_image(
        self,
        height: Optional[str] = None,
        sex: Optional[str] = None,
        weight: Optional[str] = None,
        race_ethnicity: Optional[str] = None,
        activity_level: Optional[str] = None
    ) -> Optional[dict]:
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

            # Append data row to CSV file (header is already ensured by initialization)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    rgb_path,
                    depth_path,
                    weight or '',
                    race_ethnicity or '',
                    activity_level or '',
                    height or '',
                    sex or '',
                    '',  # processing_time_seconds (will be filled after processing)
                    '',  # volume_cm3 (will be filled after processing)
                    ''   # estimated_weight_kg (will be filled after processing)
                ])

            return {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "timestamp": timestamp,
                "capture_time": time.time()  # Store capture time for processing duration calculation
            }
    
    def update_csv_with_processing_results(
        self,
        rgb_path: str,
        processing_time_seconds: float,
        volume_cm3: float,
        estimated_weight_kg: float
    ) -> bool:
        """
        Update the CSV row for a given RGB path with processing results.
        Returns True if update was successful, False otherwise.
        """
        try:
            # Read all rows
            rows = []
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows.append(header)
                for row in reader:
                    rows.append(row)
            
            # Find and update the matching row
            updated = False
            rgb_path_normalized = os.path.normpath(rgb_path)
            
            for i, row in enumerate(rows[1:], start=1):  # Skip header
                if len(row) > 0:
                    row_rgb_path = os.path.normpath(row[0])
                    if row_rgb_path == rgb_path_normalized:
                        # Ensure row has enough columns
                        while len(row) < len(rows[0]):
                            row.append('')
                        
                        # Update processing columns (indices 7, 8, 9)
                        if len(row) >= 10:
                            row[7] = f"{processing_time_seconds:.2f}"
                            row[8] = f"{volume_cm3:.2f}"
                            row[9] = f"{estimated_weight_kg:.2f}"
                        updated = True
                        break
            
            if updated:
                # Write all rows back
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                return True
            else:
                print(f"Warning: Could not find CSV row for RGB path: {rgb_path}")
                return False
        except Exception as e:
            print(f"Error updating CSV with processing results: {e}")
            import traceback
            traceback.print_exc()
            return False