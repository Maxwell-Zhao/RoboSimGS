import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import argparse
import open3d as o3d
import numpy as np

def load_pc(file_path, visualize=False):
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"error")
    
    points = np.asarray(pcd.points)
    
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    
    return points

def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=100, tolerance=0.001):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

def create_transformed_point_cloud(n_points=100, scale=1.0, rotation=0.3, translation=(0.5, 0.7)):
    A = np.random.rand(n_points, 2) * scale
    
    theta = rotation 
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    t = np.array(translation)
    B = (R @ A.T).T + t
    # B += np.random.normal(0, 0.02, B.shape)
    
    return A, B, R, t

def visualize_point_clouds(A, B, T=None, title="Point Clouds"):
    plt.figure(figsize=(10, 8))
    
    plt.scatter(A[:, 0], A[:, 1], c='blue', label='Source (A)', alpha=0.6)
    plt.scatter(B[:, 0], B[:, 1], c='red', label='Target (B)', alpha=0.6)
    
    if T is not None:
        # 将点云转换为齐次坐标
        A_homo = np.hstack([A, np.ones((A.shape[0], 1))])
        transformed_A = (T @ A_homo.T).T[:, :2]
        plt.scatter(transformed_A[:, 0], transformed_A[:, 1], 
                    c='green', label='Transformed A', alpha=0.6)
    
    plt.axis('equal')
    plt.legend()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def visualize_point_clouds_o3d(A, B, T=None, window_name="Point Clouds"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1000, height=800)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)
    
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(A)
    pcd_A.paint_uniform_color([0, 0, 1])
    vis.add_geometry(pcd_A)
    
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(B)
    pcd_B.paint_uniform_color([1, 0, 0]) 
    vis.add_geometry(pcd_B)
    
    if T is not None:
        transform_matrix = np.eye(4)
        transform_matrix[:3, :4] = T[:3, :] if T.shape == (4, 4) else T
        
        pcd_T = o3d.geometry.PointCloud()
        pcd_T.points = pcd_A.points  
        pcd_T.transform(transform_matrix)  
        pcd_T.paint_uniform_color([0, 1, 0]) 
        vis.add_geometry(pcd_T)
    
    render_opt = vis.get_render_option()
    render_opt.point_size = 3
    render_opt.background_color = np.asarray([0.9, 0.9, 0.9])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Articulated Object Segmentation and URDF Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    parser.add_argument("lerobot_real", help="Path to the real robot point cloud",default='')
    parser.add_argument("lerobot_sim", help="Path to the simulated robot point cloud",default='')
    parser.add_argument("size", help="Path to the simulated robot point cloud",default='')

    args = parser.parse_args()

    lerobot_real = load_pc(args.lerobot_real, visualize=False)
    lerobot_sim = load_pc(args.lerobot_sim, visualize=False)

    if lerobot_sim.shape[0] > lerobot_real.shape[0]:
        lerobot_sim = lerobot_sim[np.random.choice(len(lerobot_sim), lerobot_real.shape[0], replace=False)]
    else:
        lerobot_real = lerobot_real[np.random.choice(len(lerobot_real), lerobot_sim.shape[0], replace=False)]

    scale_factor = args.size
    lerobot_sim = lerobot_sim * scale_factor

    # blue A, Red B
    visualize_point_clouds_o3d(lerobot_sim, lerobot_real, window_name="Original Point Clouds")

    T_icp, distances, iterations = icp(lerobot_sim, lerobot_real, max_iterations=500, tolerance=0.00001)
    
    print(f"ICP converged after {iterations} iterations.")
    print(f"Final transformation matrix:\n{T_icp}")

    R_est = T_icp[:3, :3]
    t_est = T_icp[:3, 3]

    print(f"Estimated rotation matrix:\n{R_est}")
    print(f"Estimated translation vector: {t_est}")
    print(f"\nEvaluation Metrics:")
    print(f"Mean nearest neighbor distance (fitness score): {np.mean(distances):.4f}")
    # blue A, Red B, Green A'
    visualize_point_clouds_o3d(lerobot_sim, lerobot_real, T_icp, window_name="ICP Registration Result")






