import open3d as o3d
import numpy as np
import os
from  modelling import utils

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

def main(mesh, voxel_size: float = 0.005):
    mesh.compute_vertex_normals()
    mesh = utils.clean_mesh(mesh)
    print("Mesh cleaning done.")
    print(f"num veriticies: {len(mesh.vertices)}, num triangles: {len(mesh.triangles)}")

    # analytic (flux) volume
    analytic_vol = mesh_volume(mesh)

    print("\nVOLUME ESTIMATION RESULTS:")
    print(f"Analytic mesh volume: {analytic_vol:.6f} m³  ({(analytic_vol*1000):.2f} L)")
    return analytic_vol*1000  # return in liters