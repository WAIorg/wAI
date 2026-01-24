import os
import glob
import sys
import io
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple, Generator
from contextlib import redirect_stdout, redirect_stderr

# Add 3D-processing to path so we can import it
# From backend/app/services/processing.py, go up 3 levels to reach repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSING_DIR = REPO_ROOT / "3D-processing"
if str(PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESSING_DIR))

# Import main function from 3D-processing module
# We need to import it dynamically to avoid import errors at module load time
import importlib.util

def get_processing_pipeline():
    """Lazy import of the processing pipeline to avoid import errors."""
    main_py_path = PROCESSING_DIR / "main.py"
    if not main_py_path.exists():
        raise ImportError(
            f"main.py not found at {main_py_path}. "
            f"REPO_ROOT: {REPO_ROOT}, PROCESSING_DIR: {PROCESSING_DIR}"
        )
    
    spec = importlib.util.spec_from_file_location(
        "processing_main",
        main_py_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for main.py at {main_py_path}")
    
    module = importlib.util.module_from_spec(spec)
    # Add the processing directory to sys.path for the module's imports
    original_path = sys.path.copy()
    try:
        if str(PROCESSING_DIR) not in sys.path:
            sys.path.insert(0, str(PROCESSING_DIR))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    
    if not hasattr(module, 'main'):
        raise ImportError(f"main.py does not have a 'main' function")
    
    return module.main

# Store the function reference
_run_processing_pipeline = None

def run_processing_pipeline(*args, **kwargs):
    """Wrapper that lazily loads the processing pipeline."""
    global _run_processing_pipeline
    if _run_processing_pipeline is None:
        _run_processing_pipeline = get_processing_pipeline()
    return _run_processing_pipeline(*args, **kwargs)


def get_most_recent_capture(data_dir: str = "data") -> Optional[Tuple[str, str]]:
    """
    Get the most recent RGB image and depth array from the data folder.
    Returns (rgb_path, depth_path) tuple or None if not found.
    """
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    
    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        return None
    
    # Get all RGB files
    rgb_files = glob.glob(os.path.join(rgb_dir, "*_rgb.png"))
    if not rgb_files:
        return None
    
    # Sort by modification time, get most recent
    most_recent_rgb = max(rgb_files, key=os.path.getmtime)
    
    # Extract timestamp from RGB filename
    rgb_basename = os.path.basename(most_recent_rgb)
    timestamp = rgb_basename.replace("_rgb.png", "")
    
    # Find corresponding depth file
    depth_file = os.path.join(depth_dir, f"{timestamp}_depth.npy")
    
    if not os.path.exists(depth_file):
        return None
    
    return (most_recent_rgb, depth_file)


def get_metadata_from_csv(data_dir: str = "data", rgb_path: str = None) -> dict:
    """
    Get metadata (height, sex) from CSV for a given RGB path.
    Returns dict with height and sex, or None if not found.
    """
    import csv
    
    csv_path = os.path.join(data_dir, "captures.csv")
    if not os.path.exists(csv_path):
        return {}
    
    # Normalize paths for comparison
    if rgb_path:
        rgb_path_normalized = os.path.normpath(rgb_path)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_rgb_path = os.path.normpath(row['rgb_path'])
            if rgb_path and row_rgb_path == rgb_path_normalized:
                # Extract numeric height value (remove unit)
                height_str = row.get('height', '').strip()
                height_value = None
                if height_str:
                    # Try to extract number from "170 cm" or "170"
                    try:
                        height_value = float(height_str.split()[0])
                    except (ValueError, IndexError):
                        pass
                
                return {
                    'height': height_value,
                    'sex': row.get('sex', '').strip()
                }
    
    return {}


class LogCapture:
    """Capture stdout/stderr and yield lines in real-time."""
    def __init__(self, callback=None):
        self.logs = []
        self.lock = threading.Lock()
        self.callback = callback
    
    def write(self, text):
        with self.lock:
            self.logs.append(text)
            if self.callback:
                # Call callback with new text for streaming
                self.callback(text)
    
    def flush(self):
        pass
    
    def get_logs(self):
        with self.lock:
            return ''.join(self.logs)


def run_3d_processing(
    rgb_path: str = None,
    depth_path: str = None,
    sex: str = None,
    height: float = None,
    use_most_recent: bool = True,
    log_capture: Optional[LogCapture] = None
) -> dict:
    """
    Run the 3D processing pipeline.
    If use_most_recent is True, uses the most recent capture from data folder.
    Otherwise uses provided paths.
    """
    if use_most_recent:
        result = get_most_recent_capture()
        if result is None:
            return {"error": "No recent captures found in data folder"}
        rgb_path, depth_path = result
        
        # Try to get metadata from CSV
        metadata = get_metadata_from_csv(rgb_path=rgb_path)
        if not sex:
            sex = metadata.get('sex')
        if not height:
            height = metadata.get('height')
    
    if not rgb_path or not depth_path:
        return {"error": "RGB and depth paths are required"}
    
    if not os.path.exists(rgb_path):
        return {"error": f"RGB image not found: {rgb_path}"}
    
    if not os.path.exists(depth_path):
        return {"error": f"Depth array not found: {depth_path}"}
    
    try:
        # Normalize sex value to match weight formula expectations
        sex_normalized = None
        if sex:
            sex_lower = sex.lower()
            if sex_lower == "female":
                sex_normalized = "Female"
            elif sex_lower == "male":
                sex_normalized = "Male"
            else:
                sex_normalized = sex  # Keep original if not recognized
        
        # Capture stdout and stderr
        if log_capture:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = log_capture
            sys.stderr = log_capture
        
        try:
            # Run the processing pipeline
            result = run_processing_pipeline(
                rgb_path=rgb_path,
                depth_path=depth_path,
                sex=sex_normalized,
                height=height,
                visualize=False,
                save=True
            )
        finally:
            # Restore stdout/stderr
            if log_capture:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        return {
            "success": True,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            **result
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        if log_capture:
            log_capture.write(f"ERROR: {str(e)}\n{error_trace}")
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace
        }


def run_3d_processing_streaming(
    rgb_path: str = None,
    depth_path: str = None,
    sex: str = None,
    height: float = None,
    use_most_recent: bool = True
) -> Generator[str, None, None]:
    """
    Run 3D processing with streaming logs.
    Yields log lines and final result as JSON.
    """
    import queue
    import json
    
    log_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def log_callback(text):
        """Callback to receive log output"""
        log_queue.put(text)
    
    log_capture = LogCapture(callback=log_callback)
    
    # Run processing in a thread to capture logs
    def run_processing():
        try:
            result = run_3d_processing(
                rgb_path=rgb_path,
                depth_path=depth_path,
                sex=sex,
                height=height,
                use_most_recent=use_most_recent,
                log_capture=log_capture
            )
            result_queue.put(result)
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})
    
    # Start processing thread
    processing_thread = threading.Thread(target=run_processing, daemon=True)
    processing_thread.start()
    
    # Stream logs and wait for result
    result = None
    buffer = ""
    
    while result is None or processing_thread.is_alive():
        # Check for new logs
        try:
            while True:
                log_text = log_queue.get_nowait()
                buffer += log_text
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        yield f"data: {line}\n\n"
        except queue.Empty:
            pass
        
        # Check for result
        try:
            result = result_queue.get_nowait()
            break
        except queue.Empty:
            pass
        
        # Small delay to avoid busy waiting
        import time
        time.sleep(0.05)
    
    # Process any remaining buffer
    if buffer.strip():
        for line in buffer.split('\n'):
            if line.strip():
                yield f"data: {line}\n\n"
    
    # Wait for thread to finish
    processing_thread.join(timeout=2)
    
    # If we didn't get result yet, try one more time
    if result is None:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = {"success": False, "error": "Processing timed out"}
    
    # Yield final result
    yield f"data: RESULT:{json.dumps(result)}\n\n"
