"""
Run full wAI modelling pipeline:
1) runs SAM3D
2) Does the registration
3) Returns watertight mesh
"""

import os
from pathlib import Path
import open3d as o3d
import numpy as np
from modelling import utils
import yaml
import sys
import torch

SAM3D_ROOT = "C:/Users/wai/wAI/3D-processing/modelling/sam-3d-body"
sys.path.insert(0, SAM3D_ROOT)
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

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


def init_sam3d(sam3d_ckpt, mhr_path, device):
    model, model_cfg = load_sam_3d_body(sam3d_ckpt, device=device, mhr_path=mhr_path)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None, #use the yolo box
        human_segmentor=None, #dont need to segment the person
        fov_estimator=None, #dont need moge2
    )
    print("Returned the sam3d estimator")
    return estimator


def create_pose_sam3d(img, x1, y1, x2, y2,config, estimator, device, visualize: bool = False): 
    K = np.array([
        [config["camera"]["fx"], 0.0,config["camera"]["cx"]],
        [0.0,config["camera"]["fy"],config["camera"]["cy"]],
        [0.0,0.0,1.0],
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

    if visualize: 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="SAM3D Vertices",
            width=800,
            height=600,
        )
    return verts

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

    sam_pcd = o3d.geometry.PointCloud()
    sam_pcd.points = o3d.utility.Vector3dVector(best.astype(np.float64))

    return sam_pcd

def preprocess_pcd(pcd, voxel=0.01):
    p = pcd.voxel_down_sample(voxel)
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=30)
    )
    return p

def pcd_icp_registration(pcd_depth, pcd_sam, save: bool = False):
    sam_pcd = preprocess_pcd(pcd_sam, 0.01)
    tgt = preprocess_pcd(pcd_depth, 0.01)

    reg = o3d.pipelines.registration.registration_icp(
        sam_pcd, tgt,
        max_correspondence_distance=0.05,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    pcd_sam.transform(reg.transformation)
    if save:
        o3d.io.write_point_cloud('./point_cloud_sam3d.ply', pcd_sam)
    return pcd_sam

def mesh_icp_registration(depth_pcd, sam_mesh):
    src = sam_mesh.sample_points_uniformly(number_of_points=30000)
    tgt = depth_pcd

    src = preprocess_pcd(src, 0.01)
    tgt = preprocess_pcd(tgt, 0.01)

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=0.05,
        init=np.eye(4),
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    mesh_aligned = o3d.geometry.TriangleMesh(sam_mesh) 
    mesh_aligned.transform(reg.transformation)
    mesh_aligned.compute_vertex_normals()
    return mesh_aligned

def get_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh = utils.clean_mesh(mesh)
    return mesh

def main(img_rgb, x1, y1, x2, y2, point_cloud, visualize: bool = True, save: bool = True, progress_callback=None):
    paths, config = load_config(CONFIG_PATH)

    # 1) Run SAM3D
    print("Initializing SAM3D estimator...")
    if progress_callback:
        progress_callback(0, "Initializing SAM3D model...")
    sam3d_estimator = init_sam3d(paths["sam3d_model_checkpoint"], paths["mhr_model_checkpoint"], device)
    
    if progress_callback:
        progress_callback(20, "Creating 3D body pose...")
    verts = create_pose_sam3d(img_rgb, x1, y1, x2, y2, config, sam3d_estimator, device)
    
    if progress_callback:
        progress_callback(40, "Aligning SAM3D model with depth data...")
    sam_pcd = prealign_best(verts, point_cloud)

    if visualize:
        point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        sam_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([point_cloud, sam_pcd])

    if save:
        o3d.io.write_point_cloud(paths["sam_pt_cloud_ply_path"], sam_pcd)
        print("SAM point cloud saved at:", paths["sam_pt_cloud_ply_path"])
    print("SAM3D pose created. Proceeding to ICP alignment...")

    # 2) Run ICP registration
    if progress_callback:
        progress_callback(60, "Running ICP registration...")
    # if we want to save sam3d pcd (not needed for pipeline)
    # pcd_icp_registration(point_cloud, sam_pcd, save=False)

    faces = np.asarray(sam3d_estimator.faces) 
    sam_mesh = get_mesh(verts, faces)
    mesh_aligned = mesh_icp_registration(point_cloud, sam_mesh)
    
    # sanity view
    if visualize:
        mesh_aligned.paint_uniform_color([1.0, 0.0, 0.0])
        point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        o3d.visualization.draw_geometries([point_cloud, mesh_aligned])

    # 3) Make watertight mesh 
    if progress_callback:
        progress_callback(80, "Creating watertight mesh...")
    watertight_mesh = utils.make_watertight_meshfix(mesh_aligned)
    
    if save:
        o3d.io.write_triangle_mesh("sam3d_mesh_aligned.ply", watertight_mesh)
    
    if progress_callback:
        progress_callback(100, "3D modelling complete")
    
    return watertight_mesh

