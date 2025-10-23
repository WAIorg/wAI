import torch, numpy as np, cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import open3d as o3d
from segment_anything import SamPredictor, sam_model_registry
import os
import subprocess

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
    
    return os.path.abspath(sam_checkpoint)

# YOLO person recognition 
def person_recognition(frame_rgb):
    
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
def person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)

    predictor = SamPredictor(sam) # SAM predictor
    predictor.set_image(img_rgb) # set image to predict on
    box = np.array([x1, y1, x2, y2]) # use the bounding box from YOLO as the input for SAM
    masks, scores, logits = predictor.predict(box=box[None, :], multimask_output=False) # only output 1 mask
    person_segmentation = masks[0] # final person segmentation

    # display
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.imshow(person_segmentation, alpha=0.5, cmap='Reds')
    plt.axis('off')
    plt.show()

    return person_segmentation

def so_no_head(img_rgb, person_segmentation):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    person_mask_cv2 = (person_segmentation.astype(np.uint8)) * 255
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    (x, y, w, h) = faces[0]
    
    padding = int(0.1 * h)
    y1 = max(0, y - padding)
    y2 = min(person_mask_cv2.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(person_mask_cv2.shape[1], x + w + padding)
    
    # create face mask
    face_mask = np.zeros(person_mask_cv2.shape, dtype=np.uint8)
    face_mask[y1:y2, x1:x2] = 1

    # subtract face from body mask
    body_without_head = person_mask_cv2.copy()
    body_without_head[face_mask == 1] = 0
    return body_without_head*255

# overlay segmentation with depth
def overlay_segmentation_with_depth(depth_img, so_no_head_mask):
    
    depth_img = np.load(depth_img)
    mask = so_no_head_mask.astype(bool)
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

    # display
    depth_vis = depth_img.copy().astype(np.float32)
    depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
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
    depth_map = np.nan_to_num(depth_map, nan=0.0) # nan for 0 value pixels
    H, W = depth_map.shape

    u = np.arange(W) # creating the pixel grid
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    z = depth_map.flatten()
    valid = z > 0 # keep valid points within depth percentiles

    v_flipped = H - 1 - v[valid] # ensures orientation is correct
    filtered_depth_mask = np.stack([u[valid], v_flipped, z[valid]], axis=-1)

    return filtered_depth_mask

# create the point cloud from depth data
def create_point_cloud(filtered_depth_mask):
    
    pcd = o3d.geometry.PointCloud() # create point cloud
    pcd.points = o3d.utility.Vector3dVector(filtered_depth_mask)
    labels = np.array(pcd.cluster_dbscan(eps=10.0, min_points=50)) # remove floating blobs
    largest_label = np.bincount(labels[labels >= 0]).argmax() # keep largest blob (person)
    person_point_cloud = pcd.select_by_index(np.where(labels == largest_label)[0])

    # visualize
    o3d.visualization.draw_geometries([person_point_cloud])

    return person_point_cloud

# run segmentation pipeline
def run_pipeline(frame_rgb, depth_arr):
    sam_checkpoint = download_sam()
    img_rgb, x1, y1, x2, y2 = person_recognition(frame_rgb)
    person_segmentation_mask = person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint)
    so_no_head_mask = so_no_head(img_rgb, person_segmentation_mask)
    depth_segmentation_mask = overlay_segmentation_with_depth(depth_arr, so_no_head_mask)
    filtered_depth_mask = filter_depth_outliers(depth_segmentation_mask)
    point_cloud = create_point_cloud(filtered_depth_mask)
    return point_cloud

if __name__ == "__main__":
    frame_rgb = "./images/sub-004/rgb_2.png"
    depth_arr = "./images/sub-004/depth_raw_2.npy"
    point_cloud = run_pipeline(frame_rgb, depth_arr)