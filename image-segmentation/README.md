# 2D Reconstruction Code Documentation

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
   > Note: You need around **2.4 GB** of space for the SAM checkpoint.

5. Change the paths to the RGB image and the depth array in the `__main__` call to point to your data.
6. Clone the SAM3D Body repoitory with 

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install decord psutil
```

7. Download the SAM3D Huggingface model from: https://huggingface.co/facebook/sam-3d-body-dinov3, and follow the instructions to request the model with your HuggingFace account in order to be granted access

8. Run the pipeline:
   ```bash
   python segmentation_script.py
   ```

---

## Function Descriptions

### `download_sam()`
**Inputs:** None  
**Outputs:** SAM checkpoint path  

**Description:**  
- Pulls the **Segment Anything Model (SAM)** checkpoint.  
- Downloads the *best-performing (and heaviest)* model.  
- Places the checkpoint at the same level as the segmentation script inside `image-segmentation`.  
- Ensures it is only downloaded once to save space.  
- The file is ignored by Git (included in `.gitignore`).  

---

###  `person_recognition(frame_rgb)`
**Inputs:** RGB frame  
**Outputs:** RGB (cv2 converted) image, 4 points for the person recognition bounding box  

**Description:**  
- Uses the **YOLOv8** model for person detection.  
- Loads and converts the input image using OpenCV.  
- Predicts the bounding box around the person.  
- Saves and displays the bounding box overlay on the image.  

---

###  `person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint)`
**Inputs:** RGB frame, bounding box coordinates, SAM checkpoint path  
**Outputs:** Person segmentation mask  

**Description:**  
- Loads the SAM checkpoint onto the device.  
- Sets the image for segmentation prediction.  
- Uses the bounding box as input and generates a single segmentation mask within it.  
- Displays the resulting segmentation.  

---

###  `overlay_segmentation_with_depth(depth_img, person_mask)`
**Inputs:** Depth numpy array, person segmentation mask  
**Outputs:** Depth mask  

**Description:**  
- Loads the depth numpy array and validates that it is not too flat or zeroed.  
- Overlays the segmentation mask on the depth map.  
- Sets depth values outside the mask to 0.  
- Displays the segmentation overlaid on the depth values.  

---

###  `filter_depth_outliers(depth_map)`
**Inputs:** Depth mask  
**Outputs:** Filtered and scaled depth map  

**Description:**  
- Uses camera intrinsics to convert raw depth values (0–2000) into distances in meters.  
- Filters outliers and invalid readings.  
- Produces a clean, scaled depth map.  

---

###  `create_point_cloud(filtered_depth_mask)`
**Inputs:** Filtered and scaled depth value map  
**Outputs:** 3D point cloud of the person  

**Description:**  
- Converts depth values to 3D point cloud coordinates.  
- Keeps only the largest connected cluster (the person).  

---

###  `run_pipeline(frame_rgb, depth_arr)`
**Inputs:** RGB image and depth array  
**Outputs:** 3D point cloud  

**Description:**  
- Calls all the above functions sequentially to execute the complete pipeline.  
- Produces a final 3D reconstruction of the person from the given RGB and depth data.  

---

## Folder Structure 

```
image-segmentation/
│
├── segmentation_script.py
├── requirements.txt
├── venv/
├── test_scripts
├── sam_vit_h.pth
├── .gitignore
├── README.md
├── yolov8n.pt
└── data/
    ├── rgb_image.png
    └── depth.npy
```

## Example Usage

- change lines 166 and 167 to point to your data
```python
if __name__ == "__main__":
    rgb_image_path = "data/rgb_image.png"
    depth_array_path = "data/depth.npy"
    
    run_pipeline(rgb_image_path, depth_array_path)
```
