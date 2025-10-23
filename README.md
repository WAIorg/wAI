# Official Repository of Wai
Kinect Data Collection Web App
==============================

Monorepo contains:
- `backend/` FastAPI service for Kinect streaming, recording, image capture, and CSV metadata
- `frontend/` React app for UI controls and live preview

Backend
-------
Prereqs: Python 3.11+, Kinect v1 with `freenect` installed (libfreenect), OpenCV.

Create venv and install:
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` basic health
- `GET /stream/rgb` MJPEG stream
- `POST /capture/record/start` starts recording both RGB and depth; returns take directory
- `POST /capture/record/stop` stops recording; saves `depth_raw.npy`
- `POST /capture/image` captures one image pair; returns image directory
- `POST /metadata` body `{ name, weight, age, sex, media_path }` appends a row to `data_collection_1/participants.csv`

Data layout mirrors the original script under `data_collection_1/`.

Frontend
--------
Prereqs: Node 18+.

Install and run:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. Configure API base via env: create `.env` with `VITE_API_BASE=http://localhost:8000` if backend runs elsewhere.

Notes
-----
- If Kinect frames do not appear, verify `freenect` and device permissions. The backend’s Kinect service will stream once frames are available.
- CSV appends on each metadata save; ensure `media_path` references the returned take/image directory.

**Authors:** Mackenzie Snyder, Alexis Bader, Adele Younis, Gavin Depiero

**Project:** Image Processing Weight Estimation for Wheelchair Users