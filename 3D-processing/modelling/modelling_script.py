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


def create_pose_sam3d(img, x1, y1, x2, y2, person_segmentation_mask, config, estimator, device, visualize: bool = False): 
    K = np.array([
        [config["camera"]["fx"], 0.0,config["camera"]["cx"]],
        [0.0,config["camera"]["fy"],config["camera"]["cy"]],
        [0.0,0.0,1.0],
    ], dtype=np.float32)

    cam_int = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
    outputs = estimator.process_one_image(
        img,                 
        bboxes=np.array([[x1,y1,x2,y2]], dtype=np.float32),
        masks=person_segmentation_mask,
        cam_int=cam_int,
        use_mask=True,      
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

def get_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh = utils.clean_mesh(mesh)
    return mesh

def main(img_rgb, x1, y1, x2, y2, point_cloud, person_segmentation_mask, visualize: bool = True, save: bool = True, progress_callback=None):
    paths, config = load_config(CONFIG_PATH)

    # 1) Run SAM3D
    print("Initializing SAM3D estimator...")
    if progress_callback:
        progress_callback(0, "Initializing SAM3D model...")
    sam3d_estimator = init_sam3d(paths["sam3d_model_checkpoint"], paths["mhr_model_checkpoint"], device)
    
    if progress_callback:
        progress_callback(20, "Creating 3D body pose...")
    verts = create_pose_sam3d(img_rgb, x1, y1, x2, y2, person_segmentation_mask, config, sam3d_estimator, device)

    faces = np.asarray(sam3d_estimator.faces) 
    sam_mesh = get_mesh(verts, faces)

    # sanity view
    if visualize:
        sam_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries(sam_mesh)

    # 3) Make watertight mesh 
    if progress_callback:
        progress_callback(80, "Creating watertight mesh...")
    watertight_mesh = utils.make_watertight_meshfix(sam_mesh)
    
    if save:
        o3d.io.write_triangle_mesh("sam3d_mesh.ply", watertight_mesh)
    
    if progress_callback:
        progress_callback(100, "3D modelling complete")
    
    return watertight_mesh

