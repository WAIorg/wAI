import open3d as o3d
import numpy as np
import os
import yaml
from  modelling import utils

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
    return paths

def mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Compute mesh volume using divergence theorem.
    Need a watertight mesh! (no holes or outliers)
    """
    triangles = np.asarray(mesh.triangles)
    vertices  = np.asarray(mesh.vertices)
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]] # vertexs of the triangle
    # einsum = Einstein sum: dot product between the cross product
    volume = np.abs(np.sum(np.einsum('ij,ij->i', np.cross(v0, v1), v2))) / 6.0  # cross product for outward vector (flux)
    return volume

def compute_voxel_volume(mesh: o3d.geometry.TriangleMesh, voxel_size=0.005) -> float:
    """
    Approximate volume with voxel grid (more tolerant to small gaps).
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    return len(voxel_grid.get_voxels()) * (voxel_size ** 3)

def main(mesh=None, mesh_path=None):
    if mesh_path:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None and mesh_path is None:
        raise ValueError("Either mesh or mesh_path must be provided.")
    
    mesh.compute_vertex_normals()
    mesh = utils.clean_mesh(mesh)
    print("Mesh cleaning done.")
    print(f"num veriticies: {len(mesh.vertices)}, num triangles: {len(mesh.triangles)}")

    # analytic (flux) volume
    analytic_vol = mesh_volume(mesh)

    print("\nVOLUME ESTIMATION RESULTS:")
    print(f"Analytic mesh volume: {analytic_vol:.6f} m³  ({(analytic_vol*1000):.2f} L)")
    return analytic_vol*1000  # return in liters

if __name__ == "__main__":
    paths = load_config(CONFIG_PATH)
    main(mesh_path='/Users/adeleyounis/Desktop/Capstone/wAI/3D-processing/modelling/output/adele_down/adele_down.obj')