from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os
import uuid
import json
import queue
import subprocess
import re
from pathlib import Path

from ..services.realsense import RealSenseService
from ..services.log_stream import get_log_manager
from .realsense_stream import get_realsense_service


router = APIRouter(prefix="/realsense_capture", tags=["realsense_capture"])


class CaptureRequest(BaseModel):
    sex: Optional[str] = None
    height: Optional[str] = None


class CaptureResponse(BaseModel):
    rgb_path: Optional[str] = None
    depth_array_path: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = False
    message: Optional[str] = None


def run_3d_processing(rgb_path: str, depth_path: str, session_id: str, sex: Optional[str] = None, height: Optional[str] = None):
    """Run 3D processing pipeline in background."""
    # Log to console first to verify task is running
    print(f"[BACKGROUND TASK] Starting 3D processing for session {session_id}")
    print(f"[BACKGROUND TASK] RGB: {rgb_path}")
    print(f"[BACKGROUND TASK] Depth: {depth_path}")
    
    log_manager = get_log_manager()
    
    # Redirect stdout/stderr to capture logs
    class LogWriter:
        def __init__(self, session_id: str, log_type: str = "log"):
            self.session_id = session_id
            self.log_type = log_type
            self.buffer = ""
        
        def write(self, text: str):
            self.buffer += text
            # Flush on newlines
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line.strip():
                    log_manager.log(self.session_id, line.strip(), self.log_type)
        
        def flush(self):
            if self.buffer.strip():
                log_manager.log(self.session_id, self.buffer.strip(), self.log_type)
                self.buffer = ""
    
    stdout_writer = LogWriter(session_id, "log")
    stderr_writer = LogWriter(session_id, "error")
    
    try:
        print(f"[BACKGROUND TASK] Log manager retrieved, sending initial log...")
        log_manager.log(session_id, "Starting 3D processing pipeline...", "log")
        print(f"[BACKGROUND TASK] Initial log sent")
        # Get the repo root
        # File is at: backend/app/routers/realsense_capture.py
        # parents[0] = backend/app/routers/
        # parents[1] = backend/app/
        # parents[2] = backend/
        # parents[3] = repo root (wAI/)
        current_file = Path(__file__).resolve()
        log_manager.log(session_id, f"Current file: {current_file}", "log")
        
        repo_root = current_file.parents[3]
        processing_dir = repo_root / "3D-processing"
        main_script = processing_dir / "main.py"
        
        log_manager.log(session_id, f"Repo root: {repo_root}", "log")
        log_manager.log(session_id, f"Processing dir: {processing_dir}", "log")
        
        if not main_script.exists():
            log_manager.log(session_id, f"Error: 3D processing script not found at {main_script}", "error")
            return
        
        # Convert sex format (male/female -> Male/Female)
        sex_formatted = None
        if sex:
            sex_formatted = sex.capitalize()
        
        # Convert height to float
        height_float = None
        if height:
            try:
                height_float = float(height)
            except ValueError:
                log_manager.log(session_id, f"Warning: Could not convert height '{height}' to float", "error")
        
        log_manager.log(session_id, f"Running 3D processing with RGB: {rgb_path}, Depth: {depth_path}", "log")
        log_manager.log(session_id, f"Sex: {sex_formatted}, Height: {height_float}cm", "log")
        
        # Run the script as a subprocess to use the correct Python environment
        import subprocess
        
        # Build command - use sys.executable to use the same Python, or find the 3D-processing Python
        # Try to find Python in the 3D-processing directory (if it has a venv)
        python_cmd = sys.executable
        
        # Check if there's a venv in 3D-processing
        venv_python = processing_dir / "venv" / "bin" / "python"
        if not venv_python.exists():
            venv_python = processing_dir / "venv" / "Scripts" / "python.exe"  # Windows
        if venv_python.exists():
            python_cmd = str(venv_python)
            log_manager.log(session_id, f"Using venv Python: {python_cmd}", "log")
        else:
            log_manager.log(session_id, f"Using system Python: {python_cmd}", "log")
            log_manager.log(session_id, "Note: Ensure 3D-processing dependencies are installed in this Python environment", "log")
        
        # Build command arguments
        cmd = [
            python_cmd,
            str(main_script),
            "--rgb", rgb_path,
            "--depth", depth_path,
        ]
        
        if sex_formatted:
            cmd.extend(["--sex", sex_formatted])
        if height_float is not None:
            cmd.extend(["--height", str(height_float)])
        
        log_manager.log(session_id, f"Executing: {' '.join(cmd)}", "log")
        
        # Run subprocess and capture output in real-time
        try:
            # Set encoding to UTF-8 with error handling for Windows
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of failing
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=env
            )
            
            # Read output line by line
            for line in process.stdout:
                if line:
                    # Clean up ANSI escape codes and progress bar characters
                    line = line.strip()
                    # Remove ANSI escape sequences
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    line = ansi_escape.sub('', line)
                    # Remove other control characters that might cause issues
                    line = ''.join(char for char in line if ord(char) >= 32 or char in '\n\r\t')
                    line = line.strip()
                    
                    if line:
                        # Determine log type based on content
                        log_type = "error" if any(keyword in line.lower() for keyword in ["error", "exception", "traceback", "failed"]) else "log"
                        log_manager.log(session_id, line, log_type)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                log_manager.log(session_id, "3D processing completed successfully", "success")
            else:
                log_manager.log(session_id, f"3D processing exited with code {return_code}", "error")
                
        except Exception as proc_error:
            log_manager.log(session_id, f"Error running subprocess: {str(proc_error)}", "error")
            import traceback
            error_trace = traceback.format_exc()
            for line in error_trace.split('\n'):
                if line.strip():
                    log_manager.log(session_id, line, "error")
            raise
            
    except Exception as e:
        log_manager.log(session_id, f"Error running 3D processing: {e}", "error")
        import traceback
        error_trace = traceback.format_exc()
        for line in error_trace.split('\n'):
            if line.strip():
                log_manager.log(session_id, line, "error")
    finally:
        # Signal end of stream
        log_manager.close_stream(session_id)


@router.post("/image", response_model=CaptureResponse)
def capture_image(
    request: CaptureRequest,
    background_tasks: BackgroundTasks,
    realsense: RealSenseService = Depends(get_realsense_service)
):
    """Capture current RGB image and depth array from RealSense camera."""
    result = realsense.capture_image(
        sex=request.sex,
        height=request.height
    )
    
    if result is None:
        return CaptureResponse(
            success=False,
            message="No frames available. Ensure camera is connected and streaming."
        )
    
    # Create a session ID for this processing run
    session_id = str(uuid.uuid4())
    log_manager = get_log_manager()
    log_manager.create_stream(session_id)
    print(f"[CAPTURE] Created session {session_id} and log stream")
    
    # Trigger 3D processing in background
    # Ensure paths are absolute
    rgb_path = os.path.abspath(result["rgb_path"])
    depth_path = os.path.abspath(result["depth_array_path"])
    print(f"[CAPTURE] Adding background task with paths: RGB={rgb_path}, Depth={depth_path}")
    background_tasks.add_task(
        run_3d_processing,
        rgb_path=rgb_path,
        depth_path=depth_path,
        session_id=session_id,
        sex=request.sex,
        height=request.height
    )
    print(f"[CAPTURE] Background task added successfully")
    
    return CaptureResponse(
        rgb_path=result["rgb_path"],
        depth_array_path=result["depth_array_path"],
        timestamp=result["timestamp"],
        session_id=session_id,
        success=True,
        message="Image captured successfully. 3D processing started in background."
    )


@router.get("/logs/{session_id}")
def stream_logs(session_id: str):
    """Stream processing logs via Server-Sent Events."""
    print(f"[SSE] Client connecting to log stream for session {session_id}")
    log_manager = get_log_manager()
    stream = log_manager.get_stream(session_id)
    
    if not stream:
        print(f"[SSE] ERROR: Session {session_id} not found in log manager")
        return {"error": "Session not found"}, 404
    
    print(f"[SSE] Session found, starting event stream")
    
    def event_generator():
        print(f"[SSE] Event generator started for session {session_id}")
        try:
            while True:
                try:
                    # Wait for log entry with timeout
                    log_entry = stream.get(timeout=1.0)
                    print(f"[SSE] Got log entry: {log_entry}")
                    
                    if log_entry is None:  # End of stream signal
                        print(f"[SSE] End of stream signal received")
                        yield f"data: {json.dumps({'type': 'end', 'message': 'Processing complete'})}\n\n"
                        break
                    
                    # Format as SSE
                    yield f"data: {json.dumps(log_entry)}\n\n"
                    
                except queue.Empty:
                    # Send keepalive (silently, don't log every second)
                    yield ": keepalive\n\n"
                except Exception as e:
                    print(f"[SSE] Error in event generator: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    break
        except Exception as e:
            print(f"[SSE] Fatal error in event generator: {e}")
            import traceback
            traceback.print_exc()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
