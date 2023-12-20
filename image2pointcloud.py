"""
Convert given image to depth and then convert the depth to point cloud using cv2 reprojection
steps:
1. read file 
2. predict the depth using MiDas model
3. reproject depth to 3d using cv2
4. create the pointcloud and save it
"""
import cv2
import torch

import numpy as np

# Q matrix - Camera parameters - Can also be found using stereoRectify
Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)


# Load a MiDas model for depth estimation
model_type = "DPT_Large"     # MiDaS v3 - Large  

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midas_transforms.dpt_transform


output_points = None
output_colors = None


filename = "kennu.png"
img = cv2.imread(filename)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply input transforms
input_batch = transform(img).to(device)

# Prediction and resize to original resolution
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)


#Get rid of points with value 0 (i.e no depth)
mask_map = depth_map > 0.1

#Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = img[mask_map]

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

depth_map = (depth_map*255).astype(np.uint8)
depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

cv2.imshow('Depth Map', depth_map)




# --------------------- Create The Point Clouds ----------------------------------------

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')
 

output_file = 'pointCloudDeepLearning.ply'

create_output(output_points, output_colors, output_file)

