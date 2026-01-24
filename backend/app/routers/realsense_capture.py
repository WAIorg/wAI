from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from ..services.realsense import RealSenseService
from ..services.processing import run_3d_processing
from .realsense_stream import get_realsense_service


router = APIRouter(prefix="/realsense_capture", tags=["realsense_capture"])


class CaptureRequest(BaseModel):
    height: Optional[str] = None
    sex: Optional[str] = None
    process_after_capture: bool = False


class ImageResponse(BaseModel):
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    timestamp: Optional[str] = None
    success: bool = False
    processing_result: Optional[dict] = None


@router.post("/image", response_model=ImageResponse)
def capture_image(
    request: CaptureRequest,
    realsense: RealSenseService = Depends(get_realsense_service)
):
    """Capture RGB image and depth array from RealSense camera with metadata."""
    result = realsense.capture_image(height=request.height, sex=request.sex)
    if result is None:
        return ImageResponse(success=False)
    
    response = ImageResponse(
        rgb_path=result["rgb_path"],
        depth_path=result["depth_path"],
        timestamp=result["timestamp"],
        success=True
    )
    
    # Optionally run 3D processing after capture
    if request.process_after_capture:
        # Extract numeric height if provided
        height_value = None
        if request.height:
            try:
                height_value = float(request.height.split()[0])
            except (ValueError, IndexError):
                pass
        
        processing_result = run_3d_processing(
            rgb_path=result["rgb_path"],
            depth_path=result["depth_path"],
            sex=request.sex,
            height=height_value,
            use_most_recent=False
        )
        response.processing_result = processing_result
    
    return response
