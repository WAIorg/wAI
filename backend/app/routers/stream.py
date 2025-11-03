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
        """Stream the latest unregistered RGB frames."""
        def frame_generator():
            while True:
                rgb, _, _ = kinect.get_latest()
                if rgb is None:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nwaiting\r\n"
                    continue
                ok, jpg = cv2.imencode(".jpg", rgb)
                if not ok:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

        return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/depth")
def stream_depth(kinect: KinectService = Depends(get_kinect_service)):
        """Stream the latest depth frames (colorized for display)."""
        def frame_generator():
            while True:
                _, depth, _ = kinect.get_latest()
                if depth is None:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nwaiting\r\n"
                    continue
                depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                ok, jpg = cv2.imencode(".jpg", depth_colored)
                if not ok:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

        return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/aligned")
def stream_aligned(kinect: KinectService = Depends(get_kinect_service)):
    def frame_generator():
        while True:
            verification_img = kinect.create_alignment_verification_image()
            if verification_img is None:
                yield b"--frame\r\nContent-Type: text/plain\r\n\r\nwaiting\r\n"
                continue
            ok, jpg = cv2.imencode(".jpg", verification_img)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/verify-alignment")
def verify_alignment(kinect: KinectService = Depends(get_kinect_service)):
    """Get alignment verification metrics."""
    return kinect.verify_alignment_accuracy()


