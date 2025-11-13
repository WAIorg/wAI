import numpy as np
import open3d as o3d

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




