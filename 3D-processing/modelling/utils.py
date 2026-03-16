from logging import config
import os
import numpy as np
import open3d as o3d
import yaml
import trimesh 
import pymeshfix
from pathlib import Path

# -------- Mesh Utils ----------------

def clean_mesh(mesh: o3d.geometry.TriangleMesh):
    # remove any individual verticies or triangle of verticies
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if hasattr(mesh, "fill_holes"):
        print("filling mesh holes")
        mesh.fill_holes()
        
    return mesh

def make_watertight_meshfix(mesh_o3d: o3d.geometry.TriangleMesh,
                            keep_largest_component: bool = True) -> o3d.geometry.TriangleMesh:
    # Pre-clean in Open3D
    m = o3d.geometry.TriangleMesh(mesh_o3d)
    m = clean_mesh(m)

    tm = o3d_to_trimesh(m)

    # Optional: keep only the main body component before repair
    if keep_largest_component:
        comps = tm.split(only_watertight=False)
        if len(comps) > 1:
            tm = max(comps, key=lambda c: c.area)

    # MeshFix repair
    mf = pymeshfix.MeshFix(tm.vertices, tm.faces)
    mf.repair(verbose=False, joincomp=True, remove_smallest_components=True)

    v_fixed, f_fixed = mf.v, mf.f
    tm_fixed = trimesh.Trimesh(vertices=v_fixed, faces=f_fixed, process=False)

    # Ensure normals/orientation consistent for volume
    tm_fixed.fix_normals()
    return trimesh_to_o3d(tm_fixed)

def o3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    v = np.asarray(mesh_o3d.vertices)
    f = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)

def trimesh_to_o3d(mesh_tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_tm.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_tm.faces))
    mesh_o3d = clean_mesh(mesh_o3d)
    return mesh_o3d


