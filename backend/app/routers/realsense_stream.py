from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import cv2

from ..services.realsense import RealSenseService


def get_realsense_service() -> RealSenseService:
    # Lazy singleton attached to module
    global _realsense_service
    try:
        svc = _realsense_service
    except NameError:
        svc = RealSenseService(base_dir="data")
        try:
            svc.start()
        except Exception as e:
            print(f"Failed to start RealSense service: {e}")
            # Still return the service, but it won't be running
        _realsense_service = svc
    return svc


router = APIRouter(prefix="/realsense_stream", tags=["realsense_stream"])


@router.get("/status")
def get_status(realsense: RealSenseService = Depends(get_realsense_service)):
    """Get RealSense camera status."""
    return {
        "running": realsense.is_running(),
        "has_frames": realsense.has_frames()
    }


@router.get("/rgb")
def stream_rgb(realsense: RealSenseService = Depends(get_realsense_service)):
    """MJPEG multipart stream of RGB frames from RealSense camera."""
    import time
    
    def frame_generator():
        frames_waited = 0
        max_wait_frames = 300  # Wait up to 10 seconds for first frame
        
        while True:
            rgb, _ = realsense.get_latest()
            if rgb is None:
                frames_waited += 1
                if frames_waited > max_wait_frames:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: text/plain\r\n\r\n"
                        b"Camera not ready. Please check connection.\r\n"
                    )
                    break
                # Wait a bit before checking again
                time.sleep(0.033)  # ~30fps check rate
                continue
            
            # Reset wait counter on successful frame
            frames_waited = 0
            
            ok, jpg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                time.sleep(0.033)
                continue
            data = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )
            # Small delay to prevent overwhelming the client
            time.sleep(0.033)  # ~30fps

    return StreamingResponse(
        content=frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
