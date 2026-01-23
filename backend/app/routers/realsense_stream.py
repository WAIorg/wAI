from fastapi import APIRouter, Depends
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
        svc.start()
        _realsense_service = svc
    return svc


router = APIRouter(prefix="/realsense_stream", tags=["realsense_stream"])


@router.get("/rgb")
def stream_rgb(realsense: RealSenseService = Depends(get_realsense_service)):
    """Stream RGB frames from RealSense camera as MJPEG."""
    def frame_generator():
        while True:
            rgb = realsense.get_latest_rgb()
            if rgb is None:
                # No frame yet
                yield b"--frame\r\nContent-Type: text/plain\r\n\r\nwaiting\r\n"
                continue
            ok, jpg = cv2.imencode(".jpg", rgb)
            if not ok:
                continue
            data = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )

    return StreamingResponse(
        content=frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
