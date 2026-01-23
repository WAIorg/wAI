# 3D Processing Code Documentation

##  How to Run the Pipeline

1. **Use Python 3.11** — install it if you do not already have it on your system.
2. In the `image-segmentation` folder, create a virtual environment:
   ```bash
   python3.11 -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Install the pipeline requirements:
   ```bash
   pip install -r requirements.txt
   ```
   > Note: You need around **2.4 GB** of space for the SAM checkpoint, and **2.81 GB** for the SAM3D checkpoint

5. Change the paths to the RGB image and the depth array in the `__main__` call to point to your data.

6. Clone the SAM3D Body repoitory with 

```bash
cd modelling
git clone https://github.com/facebookresearch/sam-3d-body.git
```

7. Download the SAM3D Huggingface model from: https://huggingface.co/facebook/sam-3d-body-dinov3, and follow the instructions to request the model with your HuggingFace account in order to be granted access

8. Update the `.\config.yaml` file with the correct paths for:
-  `rgb_img_path` and `depth_img_path`, these will be your model inputs. 
- `sam3d_model_checkpoint` and `mhr_model_checkpoint` to point to the installed `sam-3d` models
- Optional: If you want to save your point clouds update `sam_pt_cloud_ply_path` and `pt_cloud_ply_path`

9. Update the `SAM3D_ROOT` file path in `modelling/modelling_script.py`

10. Run the pipeline:
   ```bash
   python main.py
   ```

## Folder Structure 

```
3D-processing/
│
├── modelling/
│  └── archives/
│  └── modelling_script.py
│  └── obj_to_volume.py
│  └── same-3d/ (follow download instructions above)
├── segmentation/
│  └── archives/
│  └── segmentation_script.py
│  └── sam_vit_h.pth (after checkpoints installed)
│  └── yoloc8n.pt (after checkpoints installed)
├── weight_calculation/
│  └── archives/
│  └── weight_formula.py
├── main.py
└── README.md
```

