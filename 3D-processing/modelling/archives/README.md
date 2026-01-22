This folder contains the code and models need to generate 3D mesh estimations from the 2D point clouds

# Setup and Installation

# 1. SMPL-x

1. Clone the SMPL-X repository:
   ```bash
   git clone https://github.com/vchoutas/smplx.git
   cd smplx
   pip install -e .

2. Download the SMPL-X model files from:
 https://smpl-x.is.tue.mpg.de/

3. Place the downloaded models here:
 ```bash
    volume-model/models/smplx/
    ├── SMPLX_FEMALE.npz
    ├── SMPLX_MALE.npz
    └── SMPLX_NEUTRAL.npz 
```

4. Verify installation:
 ```bash
python -m smplx
```

# 2. PIXIE
1. Clone the PIXIE repository:
   ```bash
   git https://github.com/yfeng95/PIXIE.git
   cd PIXIE
   pip install -e .

2. Download the PIXIE file from their README and place in volume-model/models/PIXIE

3. Download all the missing files from our capstone drive R&D/SoftwarePackages/  and place in volume-model/models/PIXIE

4. Test a sample run:

  ```bash
python demos/demo_fit_body.py -i ./data/sample_image.png --saveObj True 
```

# 3. VPoser
1. Clone the human-body-prior repository:
  ```bash
    git clone https://github.com/nghorbani/human_body_prior.git
    cd human_body_prior
    pip install -e .
```

2. Download pretrained VPoser model files from 
    https://smpl-x.is.tue.mpg.de/download.php

    Log in (free) and download VPoser v1.0 (for SMPL-X)

3. Place the downloaded VPoser model here:
  ```bash
    volume-model/models/vposer_v1_0/
    ├── snapshot/
    └── trained_model.pth
```

4. Verify installation:
    python -c "from human_body_prior.tools.model_loader import load_vposer; load_vposer('./volume-model/models/vposer')"

# 4. Other requirements
To make sure you have all the other reuirements needed within this folder run:
```bash
pip install -r requirements.txt
```

# Code Walkthrough 
- custom_pose_to_mesh.ipynb: uses SMPLX to do the custom mapping and training for our poses. This will eventually be used to map to our front view point cloud to give us accurate back and side views

- point_cloud_to_mesh.ipynb: uses our point cloud from the 2D reconstruction to generate a mesh of our front view

- pt_cloud_utils: simple util functions to aid in manipulating our point clouds

- custom_openpose.py: pulls from human_pose_estimator's repo, add's code to save our json files, and map keypoints to known poses.

- run_pixie.py: sets up and runs pixie demo model on input rgb images


