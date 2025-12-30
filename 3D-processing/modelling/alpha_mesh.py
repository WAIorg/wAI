import numpy as np
import open3d as o3d
from plyfile import PlyData 
import os
import yaml
from modelling import utils

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

def read_ply_as_o3d(paths, config, visualize=False) -> o3d.geometry.PointCloud:
    paths, config = load_config(CONFIG_PATH)

    input_path = paths["pt_cloud_ply_path"]

    plydata = PlyData.read(input_path)  
    data = np.array([list(x) for x in plydata.elements[0].data])
    points = data[:,:3]

    pcd = o3d.geometry.PointCloud()

    # add points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)

    # downsample so faster
    downpcd = pcd.voxel_down_sample(voxel_size=config["downsampling"]["voxel_size"])

    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=config["downsampling"]["normal_radius"], max_nn=config["downsampling"]["normal_max_nn"]))

    downpcd.orient_normals_to_align_with_direction()

    if visualize:
        o3d.visualization.draw_geometries([downpcd])

    print(f"Down sample shape: {len(downpcd.points)}")

    return downpcd

def inflate_and_smooth(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd_inflated = utils.inflate_point_cloud(pcd)

    pcd_smooth = utils.clean_smooth_point_cloud(pcd_inflated)

    print(f"Post-cleaning point cloud shape: {len(pcd_smooth.points)}")
    return pcd_smooth

def create_save_alpha_mesh(pcd: o3d.geometry.PointCloud, visualize=False, save=True) -> o3d.geometry.TriangleMesh:
    paths, config = load_config(CONFIG_PATH)
    alpha = config["mesh"]["alpha"]  # smaller = tighter to data, larger = smoother
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    if save:
        path = "./output/alpha_mesh.obj"
        o3d.io.write_triangle_mesh(path, mesh)
        print(f"Saved alpha mesh to {path}")

    return mesh

def main(visualize=False, save=True):
    paths, config = load_config(CONFIG_PATH)

    pcd = read_ply_as_o3d(paths, config, visualize=False)

    pcd_processed = inflate_and_smooth(pcd)

    mesh = create_save_alpha_mesh(pcd_processed, visualize=visualize, save=save)

    return mesh

