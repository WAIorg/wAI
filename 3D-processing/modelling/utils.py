from logging import config
import os
import numpy as np
import open3d as o3d
import yaml

CONFIG_PATH = "/Users/adeleyounis/Desktop/Capstone/wAI/config.yaml"

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

# -------- Point Cloud Utils ----------------
def get_cloud_length(cloud):
    vertices = np.asarray(cloud.points)
    y_coords = vertices[:, 1]
    length = y_coords.max() - y_coords.min()
    return length

def get_cloud_width(cloud):
    vertices = np.asarray(cloud.points)
    x_coords = vertices[:, 0]
    width = x_coords.max() - x_coords.min()
    return width

def get_cloud_center(cloud):
    vertices = np.asarray(cloud.points)
    center = vertices.mean(axis=0)
    return center

def stretch_vertically(cloud, scale_factor):
    center = get_cloud_center(cloud)
    vertices = np.asarray(cloud.points)
    vertices[:, 1] = (vertices[:, 1] - center[1]) * scale_factor + center[1]
    cloud.points = o3d.utility.Vector3dVector(vertices)
    return cloud

def stretch_horizontally(cloud, scale_factor):
    center = get_cloud_center(cloud)
    vertices = np.asarray(cloud.points)
    vertices[:, 0] = (vertices[:, 0] - center[0]) * scale_factor + center[0]
    cloud.points = o3d.utility.Vector3dVector(vertices)
    return cloud

def inflate_point_cloud(cloud):
    paths, config = load_config(CONFIG_PATH)

    points = np.asarray(cloud.points)

    inflate_amount = config["inflation"]["inflate_amount"]  # thicken up

    points_new = points*inflate_amount

    pcd_inflated = o3d.geometry.PointCloud()
    pcd_inflated.points = o3d.utility.Vector3dVector(points_new)

    pcd_inflated.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=config["downsampling"]["normal_radius"], 
            max_nn=config["downsampling"]["normal_max_nn"]
        )
    )
    return pcd_inflated


def clean_smooth_point_cloud(cloud):
    paths, config = load_config(CONFIG_PATH)

    # clean the pcd
    pcd_clean, ind = cloud.remove_statistical_outlier(nb_neighbors=config["cleaning"]["nb_neighbors"], 
                                                         std_ratio=config["cleaning"]["std_ratio"])

    pcd_smooth = o3d.geometry.PointCloud.voxel_down_sample(pcd_clean, voxel_size=config["cleaning"]["smooth_voxel_size"])

    pcd_smooth.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,  # adjust depending on cloud scale
        max_nn=30
        )
    )   
    return pcd_smooth


# -------- Mesh Utils ----------------

def clean_mesh(mesh: o3d.geometry.TriangleMesh):
    # remove any individual verticies or triangle of verticies
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()

    if hasattr(mesh, "fill_holes"):
        print("filling mesh holes")
        mesh.fill_holes()
        
    return mesh



