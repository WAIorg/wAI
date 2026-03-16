"""
Run full wAI segmentation pipeline:
1) Runs YOLOv8 to detect person
2) Runs SAM to segment person
3) Overlays segmentation on depth image
4) Filters depth data to create point cloud 
"""
from pathlib import Path
import torch, numpy as np, cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import open3d as o3d
from segment_anything import SamPredictor, sam_model_registry
import os
import yaml
import subprocess
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]  # adjust depth if needed
CONFIG_PATH = REPO_ROOT / "config.yaml"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_path: str):
    """Load and parse YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(rel_path):
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(base_dir, rel_path))

    # Resolve all paths
    paths = {k: resolve_path(v) for k, v in config.get("paths", {}).items()}
    return paths, config

# download SAM
def download_sam():

    # Check for existing checkpoint in 3D-processing folder first
    repo_root = Path(__file__).resolve().parents[1]  # Go up from segmentation/segmentation_script.py to 3D-processing
    possible_locations = [
        repo_root / "sam_vit_l_0b3195.pth",  # Root of 3D-processing
        repo_root / "checkpoints" / "sam_vit_l_0b3195.pth",  # checkpoints subfolder
        repo_root / "segmentation" / "sam_vit_l_0b3195.pth",  # segmentation folder
        Path("sam_vit_l_0b3195.pth"),  # Current directory (fallback)
    ]
    
    sam_checkpoint = None
    for location in possible_locations:
        if location.exists():
            sam_checkpoint = str(location.resolve())
            print(f"Found existing SAM checkpoint at: {sam_checkpoint}")
            break
    
    if sam_checkpoint is None:
        # If not found, try to download to 3D-processing root
        sam_checkpoint = str(repo_root / "sam_vit_l_0b3195.pth")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        if not os.path.exists(sam_checkpoint):
            print("Downloading SAM checkpoint...")
            urllib.request.urlretrieve(url, sam_checkpoint)
        else:
            print("SAM checkpoint already exists.")
        print(f"SAM checkpoint saved at: {os.path.abspath(sam_checkpoint)}")
    
    return os.path.abspath(sam_checkpoint), device

# YOLO person recognition 
def person_recognition(frame_rgb, visualize=False):
    # Check for existing YOLO checkpoint in 3D-processing folder first
    repo_root = Path(__file__).resolve().parents[1]  # Go up from segmentation/segmentation_script.py to 3D-processing
    possible_locations = [
        repo_root / "yolov8n.pt",  # Root of 3D-processing
        repo_root / "checkpoints" / "yolov8n.pt",  # checkpoints subfolder
        repo_root / "segmentation" / "yolov8n.pt",  # segmentation folder
    ]
    
    yolo_checkpoint = None
    for location in possible_locations:
        if location.exists():
            yolo_checkpoint = str(location.resolve())
            print(f"Found existing YOLO checkpoint at: {yolo_checkpoint}")
            break
    
    # YOLO will auto-download if not found, but prefer local checkpoint if available
    if yolo_checkpoint:
        model = YOLO(yolo_checkpoint)
    else:
        model = YOLO("yolov8n.pt")  # Will auto-download if not found
    
    print("YOLOv8 model loaded:", type(model))
    frame_rgb = cv2.imread(frame_rgb)
    if frame_rgb is None:
        raise FileNotFoundError("Image not found or cannot be opened.")
    img_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB) # convert to rgb
    preds = model.predict(source=frame_rgb, imgsz=640, conf=0.3, verbose=False)[0] # predict person
    person_boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                    int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                    float(b.conf.cpu().numpy()))
                    for b in preds.boxes if int(b.cls.cpu().numpy()) == 0]
    if not person_boxes:
        raise RuntimeError("No person detected!")

    x1, y1, x2, y2, conf = max(person_boxes, key=lambda b: b[4]) # extract box
    print(f"Person detected with confidence {conf:.2f}")
    # Note: progress callback is handled in run_pipeline

    if visualize:
        # display
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_rgb, f"person {conf:.2f}", (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        plt.figure(figsize=(8,6))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
    
    return img_rgb, x1, y1, x2, y2

# segment person from image with SAM
def person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint, visualize=False):
    sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint).to(device)

    predictor = SamPredictor(sam) # SAM predictor
    predictor.set_image(img_rgb) # set image to predict on
    box = np.array([x1, y1, x2, y2]) # use the bounding box from YOLO as the input for SAM
    masks, scores, logits = predictor.predict(box=box[None, :], multimask_output=False) # only output 1 mask
    person_segmentation = masks[0] # final person segmentation

    print(f"Segmentation mask created with shape: {person_segmentation.shape}")

    if visualize:
        # display
        plt.figure(figsize=(8,6))
        plt.imshow(img_rgb)
        plt.imshow(person_segmentation, alpha=0.5, cmap='Reds')
        plt.axis('off')
        plt.show()

    return person_segmentation

def run_pipeline(frame_rgb, config, visualize=False, save=False, progress_callback=None):
    print("Starting segmentation pipeline...")
    if progress_callback:
        progress_callback(0, "Starting segmentation pipeline...")
    
    if progress_callback:
        progress_callback(5, "Loading SAM model...")
    sam_checkpoint, device = download_sam()
    
    # 1) YOLO person recognition
    if progress_callback:
        progress_callback(10, "Detecting person in image...")
    img_rgb, x1, y1, x2, y2 = person_recognition(frame_rgb)
    
    if progress_callback:
        progress_callback(20, "Person detected, creating segmentation mask...")

    # 2) SAM person segmentation
    person_segmentation_mask = person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint, visualize=visualize)
    
    if progress_callback:
        progress_callback(50, "Segmentation complete")

    return img_rgb, person_segmentation_mask, x1, y1, x2, y2

if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    frame_rgb=paths["rgb_img_path"]
    paths, config = load_config(CONFIG_PATH)
    img_rgb, person_segmentation_mask, x1, y1, x2, y2 = run_pipeline(frame_rgb, config, True)