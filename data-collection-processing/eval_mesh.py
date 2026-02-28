import numpy as np
import trimesh
import open3d as o3d


def polyline_perimeter(points: np.ndarray) -> float:
    """Perimeter of an ordered polyline (closes last->first)."""
    if len(points) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    seg += np.linalg.norm(points[0] - points[-1])
    return float(seg)


def slice_mesh_trimesh(mesh_tm: trimesh.Trimesh, plane_origin, plane_normal):
    section = mesh_tm.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        raise ValueError("No intersection: plane does not cut the mesh.")

    # 2D slice representation + transform
    path2d, tf_3D_to_2D = section.to_2D()
    polylines2d = path2d.discrete
    if not polylines2d:
        raise ValueError("Section produced no polylines.")

    tf_2D_to_3D = np.linalg.inv(tf_3D_to_2D)

    def lift(pts2d):
        pts3_plane = np.column_stack([pts2d, np.zeros(len(pts2d))])
        return trimesh.transform_points(pts3_plane, tf_2D_to_3D)

    polylines3d = [lift(p) for p in polylines2d]
    perims = np.array([polyline_perimeter(p) for p in polylines3d])

    return polylines3d, perims


def o3d_lineset_from_polyline(points: np.ndarray, color=(1.0, 0.2, 0.2)):
    pts = o3d.utility.Vector3dVector(points)
    # connect i -> i+1 and close end -> start
    n = len(points)
    lines = [[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]]
    ls = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def o3d_plane_patch(plane_origin, plane_normal, size=0.5, color=(0.2, 0.6, 1.0)):
    # Make a square in the plane for visualization
    n = np.array(plane_normal, dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)

    # pick an arbitrary vector not parallel to n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)

    o = np.array(plane_origin, dtype=float)
    corners = np.array([
        o + size * ( u + v),
        o + size * ( u - v),
        o + size * (-u - v),
        o + size * (-u + v),
    ])

    # LineSet square
    lines = [[0,1],[1,2],[2,3],[3,0]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return ls


def view_mesh_slice_perimeter_open3d(mesh_path: str, percent_height=0.55, choose="largest"):
    # Load with trimesh for slicing
    mesh_tm = trimesh.load(mesh_path, force="mesh")
    if mesh_tm.is_empty:
        raise ValueError("Empty mesh.")

    # Print bbox scale sanity check
    ext = mesh_tm.extents
    print(f"Mesh extents (units): x={ext[0]:.3f}, y={ext[1]:.3f}, z={ext[2]:.3f}")

    # Define horizontal slice plane using bbox + centroid
    bounds = mesh_tm.bounds
    zmin, zmax = bounds[0, 2], bounds[1, 2]
    z = zmin + percent_height * (zmax - zmin)


    center = mesh_tm.bounding_box.centroid
    plane_origin = np.array([center[0], center[1], z], dtype=float)
    plane_normal = np.array([0.0, 1.0, 0.0], dtype=float)

    # Slice
    polylines3d, perims = slice_mesh_trimesh(mesh_tm, plane_origin, plane_normal)

    # Choose which loop you measure
    if choose == "largest":
        idx = int(np.argmax(perims))
        chosen = polylines3d[idx]
        perimeter = float(perims[idx])
    elif choose == "sum":
        chosen = None
        perimeter = float(perims.sum())
    else:
        raise ValueError("choose must be 'largest' or 'sum'")

    print(f"Slice z = {z:.4f}")
    print(f"Perimeter ({choose}) = {perimeter:.4f} (mesh units)")

    # Load for Open3D visualization
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    mesh_o3d.compute_vertex_normals()

    # Build slice LineSets (all loops light, chosen loop bright)
    geometries = [mesh_o3d]
    plane_ls = o3d_plane_patch(plane_origin, plane_normal, size=float(mesh_tm.extents.max()*0.35))
    geometries.append(plane_ls)

    # add all loops
    for i, pts in enumerate(polylines3d):
        if len(pts) < 3:
            continue
        color = (0.8, 0.8, 0.1)  # yellow-ish
        geometries.append(o3d_lineset_from_polyline(pts, color=color))

    # highlight chosen
    if chosen is not None:
        geometries.append(o3d_lineset_from_polyline(chosen, color=(1.0, 0.1, 0.1)))

    o3d.visualization.draw_geometries(geometries)

    return perimeter


if __name__ == "__main__":
    view_mesh_slice_perimeter_open3d(
        "20260212_115743_sam_mesh.ply",
        percent_height=0.75,
        choose="largest"
    )