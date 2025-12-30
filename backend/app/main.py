from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routers.stream import router as stream_router
from .routers.capture import router as capture_router
from .routers.metadata import router as metadata_router

app = FastAPI(title="Kinect Data Collection API")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/alignment-test")
def alignment_test():
    """Serve the alignment test HTML page."""
    return FileResponse("alignment_test.html")

app.include_router(stream_router)
app.include_router(capture_router)
app.include_router(metadata_router)


