from fastapi import APIRouter, Response
from fastapi import Depends
from typing import Callable
import cv2

from ..services.kinect import KinectService


def get_kinect_service() -> KinectService:
    # Lazy singleton attached to module
    global _kinect_service
    try:
        svc = _kinect_service
    except NameError:
        svc = KinectService(base_dir="data_collection_1")
        svc.start()
        _kinect_service = svc
    return svc


router = APIRouter(prefix="/stream", tags=["stream"])


@router.get("/rgb")
def stream_rgb(kinect: KinectService = Depends(get_kinect_service)):
    # MJPEG multipart stream
    def frame_generator():
        boundary = b"frame"
        while True:
            rgb, _ = kinect.get_latest()
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

    return Response(
        content=frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


