from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from ..services.kinect import KinectService
from .stream import get_kinect_service


router = APIRouter(prefix="/capture", tags=["capture"])


class RecordResponse(BaseModel):
    take_dir: Optional[str]


@router.post("/record/start", response_model=RecordResponse)
def record_start(kinect: KinectService = Depends(get_kinect_service)):
    take_dir = kinect.start_recording()
    return {"take_dir": take_dir}


@router.post("/record/stop", response_model=RecordResponse)
def record_stop(kinect: KinectService = Depends(get_kinect_service)):
    take_dir = kinect.stop_recording()
    return {"take_dir": take_dir}


class ImageResponse(BaseModel):
    img_dir: Optional[str]


@router.post("/image", response_model=ImageResponse)
def capture_image(kinect: KinectService = Depends(get_kinect_service)):
    img_dir = kinect.capture_image()
    return {"img_dir": img_dir}


