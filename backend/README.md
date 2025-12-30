Kinect Data Collection Web App
==============================

- `backend/` FastAPI service for Kinect streaming, recording, image capture, and CSV metadata

Backend
-------
Prereqs: Python 3.11+, Kinect v1 with `freenect` installed (libfreenect), OpenCV.

Create venv and tehn run this to install libraries:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

# Endpoints:
Streaming endpoints:
- `GET /stream/rgb` MJPEG stream
- `GET /stream/depth` Depth stream
- `GET /stream/alignment` Depth and overlayed rgb stream

Other enpoints:
- `POST /capture/record/start` starts recording both RGB and depth; returns take directory
- `POST /capture/record/stop` stops recording; saves `depth_raw.npy`
- `POST /capture/image` captures one image pair; returns image directory
- `POST /metadata` body `{ name, weight, age, sex, media_path }` appends a row to `data_collection_1/participants.csv`

# Services
kinect.py has the intrinsic values, the code for setting up the kinect fro streaming and image/video capture as well as depth-rgb alignmnet and depth calculations into meters. 

Notes
-----
- If Kinect frames do not appear, verify `freenect` and device permissions. The backend’s Kinect service will stream once frames are available.
- Data from data collection is saved in a folder called data_collection_1 in the backend folder