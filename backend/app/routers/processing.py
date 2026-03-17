from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from ..services.processing import run_3d_processing, run_3d_processing_streaming


router = APIRouter(prefix="/processing", tags=["processing"])


class ProcessingRequest(BaseModel):
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    sex: Optional[str] = None
    height: Optional[float] = None
    use_most_recent: bool = True
    capture_start_time: Optional[float] = None  # Unix timestamp when capture started


class ProcessingResponse(BaseModel):
    success: bool
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    volume: Optional[float] = None
    weight: Optional[float] = None
    std_dev_kg: Optional[float] = None
    std_dev_percent: Optional[float] = None
    sex: Optional[str] = None
    height: Optional[float] = None
    error: Optional[str] = None


@router.post("/run", response_model=ProcessingResponse)
def run_processing(request: ProcessingRequest):
    """
    Run 3D processing pipeline (non-streaming).
    If use_most_recent is True, uses the most recent capture from data folder.
    Otherwise uses provided paths.
    """
    result = run_3d_processing(
        rgb_path=request.rgb_path,
        depth_path=request.depth_path,
        sex=request.sex,
        height=request.height,
        use_most_recent=request.use_most_recent,
        capture_start_time=request.capture_start_time
    )
    
    return ProcessingResponse(**result)


@router.post("/run/stream")
def run_processing_stream(request: ProcessingRequest):
    """
    Run 3D processing pipeline with streaming logs (Server-Sent Events).
    Streams logs in real-time and sends final result as JSON.
    """
    def event_generator():
        for log_line in run_3d_processing_streaming(
            rgb_path=request.rgb_path,
            depth_path=request.depth_path,
            sex=request.sex,
            height=request.height,
            use_most_recent=request.use_most_recent,
            capture_start_time=request.capture_start_time
        ):
            yield log_line
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
