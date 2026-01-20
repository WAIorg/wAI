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
    sam_checkpoint = "sam_vit_h.pth"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print("Downloading SAM checkpoint...")
        subprocess.run(["wget", "-O", sam_checkpoint, url], check=True)
    else:
        print("SAM checkpoint already exists.")
    print(f"SAM checkpoint saved at: {os.path.abspath(sam_checkpoint)}")
    
    return os.path.abspath(sam_checkpoint), device

# YOLO person recognition 
def person_recognition(frame_rgb, visualize=False):
    
    model = YOLO("yolov8n.pt")
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
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)

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

# overlay segmentation with depth
def overlay_segmentation_with_depth(depth_img, person_mask, visualize=False):
    
    depth_img = np.load(depth_img)
    mask = person_mask.astype(bool)
    masked_depth_values = depth_img[mask] # extract depth values in the mask

    if masked_depth_values.size > 0: # compute basic depth metrics inside the mask for verification
        print("min:", float(np.nanmin(masked_depth_values)))
        print("max:", float(np.nanmax(masked_depth_values)))
        print("mean:", float(np.nanmean(masked_depth_values)))
        print("median:", float(np.nanmedian(masked_depth_values)))
    else:
        print("Mask contains no pixels (empty).")

    depth_map = np.full_like(depth_img, np.nan, dtype=np.float32) #everything not in the mask is nan
    depth_map[mask] = depth_img[mask]

    print(f"Depth map successfully overlaid")

    if visualize:
        # display
        depth_vis = depth_img.copy().astype(np.float32)
        depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_colormap)
        plt.axis('off')
        plt.title("Depth colormap")
        plt.show()
    
        overlay_color = np.zeros_like(depth_colormap) # create colored mask overlay - purpleish
        overlay_color[mask] = (255, 0, 255)                     
        alpha = 0.5                                         
        blended = depth_colormap.copy() # blend only where mask is true
        blended[mask] = cv2.addWeighted(depth_colormap[mask], 1 - alpha, overlay_color[mask], alpha, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB) # convert to rgb
        plt.figure(figsize=(8, 6))
        plt.imshow(blended_rgb)
        plt.axis('off')
        plt.title("Depth colormap with segmentation overlay (red, 50% on mask)")
        plt.show()

    return depth_map

# filter outliers 
def filter_depth_outliers(depth_map):
    fx_d, fy_d = 638.19, 638.19
    cx_d, cy_d = 639.70, 356.18
    depth_map = depth_map / 1000.0
    depth_map = np.nan_to_num(depth_map, nan=0.0) # replace nans with 0
    H, W = depth_map.shape

    u = np.arange(W) # create pixel grid
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    u_flat = u.flatten() # flatten arrays
    v_flat = v.flatten()
    z_flat = depth_map.flatten()
    valid = z_flat > 0 # keep only non-zero
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    z_valid = z_flat[valid]

    # convert pixel coordinates to metric camera coordinates
    X = (u_valid - cx_d) * z_valid / fx_d
    Y = (v_valid - cy_d) * z_valid / fy_d
    Z = z_valid
    points = np.stack([X, -Y, Z], axis=-1) # flip orienation for visuals

    return points

# create the point cloud from depth data
def create_point_cloud(filtered_depth_mask, visualize=False):
    paths, config = load_config(CONFIG_PATH)
    pcd = o3d.geometry.PointCloud() # create point cloud
    pcd.points = o3d.utility.Vector3dVector(filtered_depth_mask)
    labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=20)) # remove floating blobs
    largest_label = np.bincount(labels[labels >= 0]).argmax() # keep largest blob (person)
    person_point_cloud = pcd.select_by_index(np.where(labels == largest_label)[0])

    if visualize:
        # visualize
        o3d.visualization.draw_geometries([person_point_cloud])

    o3d.io.write_point_cloud(paths["pt_cloud_ply_path"], person_point_cloud)
    print("Point cloud saved at:", paths["pt_cloud_ply_path"])
    print("Point cloud created with shape:", np.asarray(person_point_cloud.points).shape)

    return person_point_cloud

def run_pipeline(frame_rgb, depth_arr, visualize=False):
    print("Starting segmentation pipeline...")
    
    sam_checkpoint, device = download_sam()
    # 1) YOLO person recognition
    img_rgb, x1, y1, x2, y2 = person_recognition(frame_rgb)

    # 2) SAM person segmentation
    person_segmentation_mask = person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint, visualize=visualize)
    
    # 3) Overlay segmentation with depth
    depth_segmentation_mask = overlay_segmentation_with_depth(depth_arr, person_segmentation_mask, visualize=visualize)
    filtered_depth_mask = filter_depth_outliers(depth_segmentation_mask)
    
    # 4) Create point cloud
    point_cloud = create_point_cloud(filtered_depth_mask)
    print("Finished processing point cloud")

    return point_cloud, img_rgb, x1, y1, x2, y2

if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    frame_rgb=paths["rgb_img_path"]
    depth_arr=paths["depth_img_path"]
    point_cloud = run_pipeline(frame_rgb, depth_arr, True)