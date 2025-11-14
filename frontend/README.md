Kinect Data Collection Web App
==============================

- `frontend/` React app for UI controls and live preview

Frontend
--------
Prereqs: Node 18+.

Install and run:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. 

Ensure backend server is running to run front end (see documentation in backend folder)

Notes
-----
- CSV appends on each metadata save; ensure `media_path` references the returned take/image directory.