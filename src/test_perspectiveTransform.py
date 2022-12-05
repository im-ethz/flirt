import cv2
import torch
import numpy as np


# Load the image
img_path = "/Users/philipp/deepingsource/Projects/calibrate_matrix/OFFICE_OFFICE/images/192.168.1.18.png"
img = cv2.imread(img_path)

# Load the minimap 
minimap_path = "/Users/philipp/deepingsource/Projects/calibrate_matrix/OFFICE_OFFICE/minimap/minimap_cropped.png"
minimap = cv2.imread(minimap_path)

# Load the points
points_path = "/Users/philipp/deepingsource/Projects/calibrate_matrix/OFFICE_OFFICE/points/192.168.1.11_192.168.1.12_192.168.1.17_192.168.1.18_minimap_cropped/cam.npy"
# Select folder if not selected
loaded_data = np.load(points_path, allow_pickle=True).item()

# Check if floor_points are provided
if ("point_data" in loaded_data) and ("floor_points" in loaded_data):
    # New format
    loaded_point_data = loaded_data["point_data"]
    floor_points = loaded_data["floor_points"]
else:
    # Previous format
    loaded_point_data = loaded_data

# Make sure all keys are represented as int
point_data = {}
for cam_key in loaded_point_data.keys():
    point_data[int(cam_key)] = {}
    for point_key in loaded_point_data[cam_key].keys():
        point_data[int(cam_key)][int(point_key)] = loaded_point_data[cam_key][point_key]

# Select the four matching points
img_point_ids = point_data[3].keys()
# Last index is from the minimap
minimap_point_ids = point_data[4].keys()
matching_point_ids = list(set(img_point_ids) & set(minimap_point_ids))

# PerspectiveTransform
img_points = np.array([[point_data[3][point_id] for point_id in matching_point_ids]], np.float32)
minimap_points = np.array([[point_data[4][point_id] for point_id in matching_point_ids]], np.float32)
print(img_points)
print(minimap_points)
M = cv2.getPerspectiveTransform(img_points, minimap_points)
# warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]//2))
# cv2.imshow("Img", warped)
# cv2.waitKey(0)

# Warp points onto the minimap
warped_points = cv2.perspectiveTransform(img_points, M)
print(warped_points)

# Warpperspective