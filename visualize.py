"""
visuaize point clouds 
"""

import open3d as o3d

# Load the point cloud which is created using depth2point.py
original_cloud = o3d.io.read_point_cloud("pointCloudDeepLearning.ply")

# Voxel grid downsampling
voxel_size = 0.02  # Adjust the size of the voxels as needed
downsampled_cloud = original_cloud.voxel_down_sample(voxel_size)

downsampled_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

o3d.visualization.draw_geometries([downsampled_cloud])





# o3d.io.write_point_cloud("downsampled_cloud_with_normals.pcd", downsampled_cloud)

# # Print the number of vertices in the original and downsampled clouds
# print(f"Original Point Cloud: {len(original_cloud.points)} vertices")
# print(f"Downsampled Point Cloud: {len(downsampled_cloud.points)} vertices")
