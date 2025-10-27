import numpy as np
import trimesh

def obj_to_ply(obj_path, ply_path, num_points=10000):
    mesh = trimesh.load(obj_path)
    points, face_indices = mesh.sample(num_points, return_index=True)
    normals = mesh.face_normals[face_indices]
    with open(ply_path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n") 
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("end_header\n")
        for i in range(len(points)):
            point = points[i]
            normal = normals[i]
            ply_file.write(f"{point[0]} {point[1]} {point[2]} "
                           f"{normal[0]} {normal[1]} {normal[2]}\n")

obj_to_ply(
    "/home/haoyu/code/RoboSimGS/assets/objects/banana/AR-Code-Object-Capture-app-1751700674.obj",
    "/home/haoyu/code/RoboSimGS/assets/objects/banana/banana_point_cloud.ply",
    num_points=10000
)