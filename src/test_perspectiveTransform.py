import cv2
import torch
import numpy as np


# Load the image
img_path = "/Users/philipp/deepingsource/Projects/calibrate_matrix/OFFICE_OFFICE/images/192.168.1.18.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

# Load the calibration data (obtained wth camera_calibration.py)
calib_param_path = "/Users/philipp/deepingsource/Projects/3rd_floor_checkerboard_images/onvif_004_pete_v2/calibration.npz"
out = np.load(calib_param_path)
# ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']
mtx = out['mtx']
dist = out['dist']

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
# Exclude point 10 due to colinearity
img_points = np.array([[point_data[3][point_id] for point_id in matching_point_ids if point_id != 10]], np.float32)
# Camera calibration was performed with down-sized image
img_points = img_points // 2
minimap_points = np.array([[point_data[4][point_id] for point_id in matching_point_ids if point_id != 10]], np.float32)

# Undistort the image
undist_img = cv2.undistort(img, mtx, dist, None, mtx)
# Undistort the image points
undist_img_points = cv2.undistortPoints(img_points, mtx, dist, None, mtx)
undist_img_points = np.swapaxes(undist_img_points, 0, 1)

# Get the perspective transform
M = cv2.getPerspectiveTransform(undist_img_points, minimap_points)
# Warp the image 
warped_img = cv2.warpPerspective(undist_img, M, (minimap.shape[1], minimap.shape[0]))
# Perspective transform the points
warped = cv2.perspectiveTransform(undist_img_points, M)

print("#### Image points ####")
print(img_points)
print("#### Undistorted image points ####")
print(undist_img_points)
print("#### Minimap points ####")
print(minimap_points)
print("#### Warped ####")
print(warped)

points = []
# Warp points onto the minimap
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

# Plot the warped points over the minimap
for p in warped[0]:
    cv2.circle(minimap, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
cv2.imshow("Minimap + Warped points", minimap)

# Plot the warped on the warped image
for p in warped[0]:
    cv2.circle(warped_img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
cv2.imshow("Warped img", warped_img)

# Plot the warped points over the minimap
for wp in warped_points:
    for p in wp:
        cv2.circle(minimap, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
cv2.imshow("Img + Projected points", minimap)
cv2.waitKey(0)