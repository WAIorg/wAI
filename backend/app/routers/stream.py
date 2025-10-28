from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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

    return StreamingResponse(
        content=frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/depth")
def stream_depth(kinect: KinectService = Depends(get_kinect_service)):
    # MJPEG multipart stream for registered depth
    def frame_generator():
        boundary = b"frame"
        while True:
            rgb, registered_depth = kinect.get_latest_registered()
            if registered_depth is None:
                # No frame yet
                yield b"--frame\r\nContent-Type: text/plain\r\n\r\nwaiting\r\n"
                continue
            
            # Convert depth to visualization
            depth_vis = cv2.convertScaleAbs(registered_depth, alpha=0.03)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            ok, jpg = cv2.imencode(".jpg", depth_colored)
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


@router.get("/aligned")
def stream_aligned(kinect: KinectService = Depends(get_kinect_service)):
    # MJPEG multipart stream showing RGB and depth overlaid
    def frame_generator():
        boundary = b"frame"
        while True:
            verification_img = kinect.create_alignment_verification_image()
            if verification_img is None:
                # No frame yet
                yield b"--frame\r\nContent-Type: text/plain\r\n\r\nwaiting\r\n"
                continue
            
            ok, jpg = cv2.imencode(".jpg", verification_img)
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


@router.get("/verify-alignment")
def verify_alignment(kinect: KinectService = Depends(get_kinect_service)):
    """Get alignment verification metrics."""
    return kinect.verify_alignment_accuracy()


