import torch, numpy as np, cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import open3d as o3d
import sys
from segment_anything import SamPredictor, sam_model_registry
SAM3D_ROOT = "/Users/mackenziesnyder/Desktop/capstone/wAI/image-segmentation/sam-3d-body"
sys.path.insert(0, SAM3D_ROOT)
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
import os
import subprocess

# download SAM
def download_sam():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_checkpoint = "sam_vit_h.pth"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print("Downloading SAM checkpoint...")
        subprocess.run(["wget", "-O", sam_checkpoint, url], check=True)
    else:
        print("SAM checkpoint already exists.")
    print(f"SAM checkpoint saved at: {os.path.abspath(sam_checkpoint)}")
    
    return os.path.abspath(sam_checkpoint), device

def init_sam3d(sam3d_ckpt, mhr_path, device):
    model, model_cfg = load_sam_3d_body(sam3d_ckpt, device=device, mhr_path=mhr_path)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None, #use the yolo box
        human_segmentor=None, #dont need to segment the person
        fov_estimator=None, #dont need moge2
    )
    print("returned the sam3d estimator")
    return estimator

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
def person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint, device):
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

# overlay segmentation with depth
def overlay_segmentation_with_depth(depth_img, person_mask):
    
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
    
    fx_d, fy_d = 596.25827383, 593.35350108 # camera intrinsics
    cx_d, cy_d = 328.00224565, 246.72323964
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
def create_point_cloud(filtered_depth_mask):
    
    pcd = o3d.geometry.PointCloud() # create point cloud
    pcd.points = o3d.utility.Vector3dVector(filtered_depth_mask)
    labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=20)) # remove floating blobs
    largest_label = np.bincount(labels[labels >= 0]).argmax() # keep largest blob (person)
    person_point_cloud = pcd.select_by_index(np.where(labels == largest_label)[0])

    # visualize
    o3d.visualization.draw_geometries([person_point_cloud])
    o3d.io.write_point_cloud('./point_cloud.ply', person_point_cloud)

    return person_point_cloud

def create_pose_sam3d(img, x1, y1, x2, y2,estimator, device): 
    
    K = np.array([
        [596.25827383, 0.0,         328.00224565],
        [0.0,          593.35350108,246.72323964],
        [0.0,          0.0,           1.0],
    ], dtype=np.float32)

    cam_int = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
    outputs = estimator.process_one_image(
        img,                 
        bboxes=np.array([[x1,y1,x2,y2]], dtype=np.float32),
        masks=None,
        cam_int=cam_int,
        use_mask=False,      
        inference_type="body",
    )
    out0 = outputs[0]
    verts = np.asarray(out0["pred_vertices"], dtype=np.float64)
    return verts

def visualize_vertices(verts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="SAM3D Vertices",
        width=800,
        height=600,
    )
    return pcd

def bbox_extent_np(pts):
    pts = np.asarray(pts)
    return pts.max(axis=0) - pts.min(axis=0)

def make_rot(axis, deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == "y":
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    if axis == "z":
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    raise ValueError(axis)

def prealign_best(verts, depth_pcd):
    depth_pts = np.asarray(depth_pcd.points)

    # --- scale match (uniform) ---
    depth_extent = bbox_extent_np(depth_pts)
    sam_extent   = bbox_extent_np(verts)
    scale = np.linalg.norm(depth_extent) / (np.linalg.norm(sam_extent) + 1e-9)
    v0 = verts * scale

    depth_center = depth_pts.mean(axis=0)

    # candidate flips (handedness) and yaw rotations (facing)
    flips = [
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (1, -1, -1),
        (-1, -1, -1),
    ]
    yaws = [0, 90, 180, 270]  # rotate around "up" axis (we'll test around Y and Z too)

    # We'll test yaw around Y *and* Z since "up" might differ between frames
    yaw_axes = ["y", "z"]

    best = None
    best_score = np.inf
    best_meta = None

    # build a KDTree on depth for quick scoring
    depth_o3d = o3d.geometry.PointCloud()
    depth_o3d.points = o3d.utility.Vector3dVector(depth_pts.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(depth_o3d)

    def score_points(v):
        # sample to speed up scoring
        if v.shape[0] > 5000:
            idx = np.random.choice(v.shape[0], 5000, replace=False)
            vv = v[idx]
        else:
            vv = v
        # mean NN distance into depth cloud
        dsum = 0.0
        for p in vv:
            _, _, d2 = kdtree.search_knn_vector_3d(p, 1)
            dsum += float(d2[0])
        return dsum / len(vv)

    for fx, fy, fz in flips:
        vflip = v0.copy()
        vflip[:, 0] *= fx
        vflip[:, 1] *= fy
        vflip[:, 2] *= fz

        for ax in yaw_axes:
            for yaw in yaws:
                R = make_rot(ax, yaw)
                v = (vflip @ R.T)

                # center
                v += depth_center - v.mean(axis=0)

                sc = score_points(v)
                if sc < best_score:
                    best_score = sc
                    best = v
                    best_meta = (fx, fy, fz, ax, yaw, scale)

    print("Best prealign (fx,fy,fz, yaw_axis, yaw_deg, scale):", best_meta)
    print("Best score (mean NN d^2):", best_score)
    return best, best_meta

def prealign_sam3d_to_depth(verts, depth_pcd):
    depth_pts = np.asarray(depth_pcd.points)
    depth_center = depth_pts.mean(axis=0)

    # --- scale match ---
    depth_extent = bbox_extent_np(depth_pts)
    sam_extent   = bbox_extent_np(verts)
    scale = np.linalg.norm(depth_extent) / (np.linalg.norm(sam_extent) + 1e-9)
    verts2 = verts * scale

    # --- axis fix (common) ---
    verts2[:, 1] *= -1
    verts2[:, 2] *= -1

    # --- center ---
    verts2 += depth_center - verts2.mean(axis=0)

    print("Depth extent:", depth_extent)
    print("SAM extent:", sam_extent)
    print("Scale:", scale)

    return verts2

def preprocess_pcd(pcd, voxel=0.01):
    p = pcd.voxel_down_sample(voxel)
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=30)
    )
    return p

def align_rigid_icp(pcd_depth, pcd_sam, voxel=0.01, max_corr=0.05):
    src = preprocess_pcd(pcd_sam, voxel)
    tgt = preprocess_pcd(pcd_depth, voxel)

    init = np.eye(4)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_corr,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    print("ICP fitness:", reg.fitness, "rmse:", reg.inlier_rmse)
    return reg.transformation

def similarity_icp_mesh_to_depth(mesh: o3d.geometry.TriangleMesh,
                                 depth_pcd: o3d.geometry.PointCloud,
                                 voxel=0.01,
                                 max_corr=0.05,
                                 n_mesh_samples=30000):
    # sample points from mesh
    src = mesh.sample_points_uniformly(number_of_points=n_mesh_samples)
    tgt = depth_pcd

    src = preprocess_pcd(src, voxel)
    tgt = preprocess_pcd(tgt, voxel)

    # IMPORTANT: point-to-point supports scaling
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    print("ICP fitness:", reg.fitness, "rmse:", reg.inlier_rmse)
    return reg.transformation

def sam3d_mesh_from_verts_faces(verts: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh

# run segmentation pipeline
def run_pipeline(frame_rgb, depth_arr):

    sam3d_ckpt = "./sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr_path = "./sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    sam_checkpoint, device = download_sam()
    sam3d_estimator = init_sam3d(sam3d_ckpt, mhr_path, device)
    img_rgb, x1, y1, x2, y2 = person_recognition(frame_rgb)
    person_segmentation_mask = person_segmentation(img_rgb, x1, y1, x2, y2, sam_checkpoint, device)
    depth_segmentation_mask = overlay_segmentation_with_depth(depth_arr, person_segmentation_mask)
    filtered_depth_mask = filter_depth_outliers(depth_segmentation_mask)
    point_cloud = create_point_cloud(filtered_depth_mask)
    verts = create_pose_sam3d(img_rgb, x1, y1, x2, y2, sam3d_estimator, device)
    
    verts_prealigned, meta = prealign_best(verts, point_cloud)

    sam_pcd = o3d.geometry.PointCloud()
    sam_pcd.points = o3d.utility.Vector3dVector(verts_prealigned.astype(np.float64))

    # sanity view BEFORE ICP
    point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    sam_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([point_cloud, sam_pcd])

    # now ICP refinements
    T = align_rigid_icp(point_cloud, sam_pcd)
    sam_pcd.transform(T)
    o3d.visualization.draw_geometries([point_cloud, sam_pcd])
    o3d.io.write_point_cloud('./point_cloud_sam3d.ply', sam_pcd)

    faces = np.asarray(sam3d_estimator.faces)  # (F,3)
    mesh = sam3d_mesh_from_verts_faces(verts, faces)

    T = similarity_icp_mesh_to_depth(mesh, point_cloud, voxel=0.01, max_corr=0.05)

    mesh_aligned = o3d.geometry.TriangleMesh(mesh)  # copy
    mesh_aligned.transform(T)
    mesh_aligned.compute_vertex_normals()

    # sanity view
    mesh_aligned.paint_uniform_color([1.0, 0.0, 0.0])
    point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([point_cloud, mesh_aligned])

    # volume
    if not mesh_aligned.is_watertight():
        print("WARNING: mesh is not watertight; volume may be invalid.")
    else:
        volume_m3 = mesh_aligned.get_volume()
        print(f"Volume: {volume_m3:.6f} m³")
    
    o3d.io.write_triangle_mesh("sam3d_mesh_aligned.ply", mesh_aligned)
    np.save("sam3d_to_depth_T.npy", T)

    return point_cloud, verts

if __name__ == "__main__":
    frame_rgb = "./images/rgb.png"
    depth_arr = "./images/depth.npy"
    run_pipeline(frame_rgb, depth_arr)