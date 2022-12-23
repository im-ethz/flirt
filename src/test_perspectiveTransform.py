import os
import cv2
import numpy as np

from utils import to_homogeneous

data_base_path = "/Users/philipp/deepingsource/AWS_S3/deepingsource-calibration/3rd_floor_convenience_store"
# Load the image
ip = "192.168.1.18"
img_path = os.path.join(data_base_path, "images", ip + ".png")
img = cv2.imread(img_path)

# Load the calibration data (obtained wth camera_calibration.py)
calib_param_path = "/Users/philipp/deepingsource/Projects/3rd_floor_checkerboard_images/onvif_004_pete_v2/calibration.npz"
calib_param_path = os.path.join(data_base_path, "checkerboard_calibration", ip, "0", ip + "_intrinsics.npz")
out = np.load(calib_param_path)
# ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']
mtx = out['mtx']
dist = out['dist']

# Load the minimap 
blueprint_path = os.path.join(data_base_path, "blueprint.png")
blueprint = cv2.imread(blueprint_path)

# Load the points
points_path = os.path.join(data_base_path, "points/192.168.1.11_192.168.1.12_192.168.1.17_192.168.1.18_minimap_cropped/cam.npy")
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
# Last index is from the blueprint
blueprint_point_ids = point_data[4].keys()
matching_point_ids = list(set(img_point_ids) & set(blueprint_point_ids))

# PerspectiveTransform 
# Exclude point 10 due to colinearity
img_points = np.array([[point_data[3][point_id] for point_id in matching_point_ids if point_id != 10]], np.float32)
blueprint_points = np.array([[point_data[4][point_id] for point_id in matching_point_ids if point_id != 10]], np.float32)

# Undistort the image
undist_img = cv2.undistort(img, mtx, dist, None, mtx)
# Undistort the image points
undist_img_points = cv2.undistortPoints(img_points, mtx, dist, None, mtx)
undist_img_points = np.swapaxes(undist_img_points, 0, 1)

blueprint_points_3d = np.concatenate([blueprint_points[0], np.zeros_like(blueprint_points[0][:, :1])], axis=-1)

# Try calibrate camera again
_, _, _, rvecs_global, tvecs_global = cv2.calibrateCamera([blueprint_points_3d], img_points, img.shape[:2][::-1], mtx, dist)

# https://stackoverflow.com/questions/63531312/transforming-2d-image-point-to-3d-world-point-where-z-0
# Convert the rotation vectors to a 3x3 rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rvecs_global[0])

# Combine the rotation matrix with the translation vector to form the projection matrix
extrincic = np.hstack((rotation_matrix, tvecs_global[0]))
projection_mtx = mtx.dot(extrincic)

# delete the third column since z=0
projection_mtx = np.delete(projection_mtx, 2, 1)

# finding the inverse of the matrix 
inv_projection = np.linalg.inv(projection_mtx)

img_p = to_homogeneous(undist_img_points[0])

#calculating the 3D point which located on the 3D plane
point_3d = inv_projection.dot(img_p[0])

projection_matrix = projection_matrix.astype(np.float32)
transformation_matrix = projection_matrix[:, :3]


# Get the perspective transform
M = cv2.getPerspectiveTransform(undist_img_points, blueprint_points)

# Warp the image 
warped_img = cv2.warpPerspective(undist_img, M, (blueprint.shape[1], blueprint.shape[0]))
# Perspective transform the points
warped = cv2.perspectiveTransform(img_points, M)

import pdb;pdb.set_trace()
warped = cv2.perspectiveTransform(np.array([blueprint_points_3d]), projection_matrix)


points_in = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
warped_points, _ = cv2.projectPoints(img_points[0].astype(int), rvecs_global[0], tvecs_global[0], mtx, dist)

print("#### Image points ####")
print(img_points)
print("#### Undistorted image points ####")
print(undist_img_points)
print("#### blueprint points ####")
print(blueprint_points)
print("#### Warped ####")
print(warped)

points = []
# Warp points onto the blueprint
for x in range(0, img.shape[1], 20):
    for y in range(0, img.shape[0], 20):
        undist_p = cv2.undistortPoints(np.array([[x, y]], np.float32), mtx, dist, None, mtx)
        if (undist_p < 0).any():
            continue
        if undist_p[0, 0, 0] >= img.shape[1]:
            # print(undist_p)
            continue
        if undist_p[0, 0, 1] >= img.shape[0]:
            # print(undist_p)
            continue
        points.append(np.array([x, y], np.float32))

points = np.array([points], np.float32)

# Undistort the points
undist_points = cv2.undistortPoints(points, mtx, dist, None, mtx)

warped_points = cv2.perspectiveTransform(undist_points, M)

# Plotting
# Plot the image points over the image
for p in img_points[0]:
    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
for p in points[0]:
    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
cv2.imshow("Img", img)

# Plot the undistorted image points over the undistorted image
for p in undist_img_points[0]:
    cv2.circle(undist_img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
cv2.imshow("Undistorted Img", undist_img)

# Plot the warped points over the blueprint
for p in warped[0]:
    cv2.circle(blueprint, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
cv2.imshow("blueprint + Warped points", blueprint)

# Plot the warped on the warped image
for p in warped[0]:
    cv2.circle(warped_img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
cv2.imshow("Warped img", warped_img)

# Plot the warped points over the blueprint
for wp in warped_points:
    for p in wp:
        cv2.circle(blueprint, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
cv2.imshow("Img + Projected points", blueprint)
cv2.waitKey(0)

# Check if the same holds for the originally used projection function
from utils import get_calibrated_points
thetas = rotationMatrixToEulerAngles(M)


parameters = torch.from_numpy(np.load(param_filepath))

thetas = rotationMatrixToEulerAngles(M)