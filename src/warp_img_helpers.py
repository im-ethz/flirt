# Federica Spinola 2022.10.14
# helper functions to warp images or image points to top-view and vice-versa
# inputs: jeewook's calibration parameters (in his format)
# outputs: homography matrix that can be directly used to transform points and image by pre-multiplication

import numpy as np
import cv2

from cam_to_plane import thetas_to_R

def cam_to_minimap(camera_parameters, img_size):
    # Extract raw intrinsics and extrinsics from calibration parameters
    R, C, f, k = np.split(camera_parameters, [9, 12, 13])
    R = R.reshape(3,3)
    t = -R @ C.reshape(-1, 1)

    # Get normalization parameters from image size
    image_center = np.array(img_size, dtype=np.float32) / 2
    image_norm_length = np.sqrt(np.mean(image_center ** 2))

    # Construct intrinsic matrix K
    K = np.array([[f[0]*image_norm_length, 0, image_center[0]], [0, f[0]*image_norm_length, image_center[1]], [0, 0, 1]], dtype=np.float32)

    # Construct 2D extrinsic matrix m (for homography)
    m = np.concatenate((R[:, :2], t), axis=1)

    # Construct 2D projection matrix, i.e. homography matrix H (3x3)
    H = K @ m
    H_inv = np.linalg.inv(H)

    return H_inv, K, -k[0]

def minimap_to_cam(camera_parameters, img_size):
    # Extract raw intrinsics and extrinsics from calibration parameters
    R, C, f, k = np.split(camera_parameters, [9, 12, 13])
    R = R.reshape(3,3)
    t = -R @ C.reshape(-1, 1)

    # Get normalization parameters from image size
    image_center = np.array(img_size, dtype=np.float32) / 2
    image_norm_length = np.sqrt(np.mean(image_center ** 2))

    # Construct intrinsic matrix K
    K = np.array([[f[0]*image_norm_length, 0, image_center[0]], [0, f[0]*image_norm_length, image_center[1]], [0, 0, 1]], dtype=np.float32)

    # Construct 2D extrinsic matrix m (for homography)
    m = np.concatenate((R[:, :2], t), axis=1)

    # Construct 2D projection matrix, i.e. homography matrix H (3x3)
    H = K @ m

    return H, K, -k[0]

if __name__ == "__main__":
    # test functions
    cam_id = 1
    map = cv2.imread('./data/Office/minimap.png')
    cam = cv2.imread(f'./data/Office/images/{cam_id}.png')
    npy_file_path = f'./data/Office/cam_parameters/{cam_id}.npy'

    # load camera calibrations
    camera_parameters  = np.load(npy_file_path)
    camera_parameters = np.concatenate(
                        [thetas_to_R(camera_parameters[:3], True).reshape(-1),
                        camera_parameters[3:]],axis=0) # convert rotations from euler to rotation matrix
    
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    # Get cam to minimap homography matrix
    H_cam2minimap, K, distCoef = cam_to_minimap(camera_parameters, (cam.shape[1], cam.shape[1]))

    # Undistort image
    cam_undistort = cv2.undistort(cam, K, np.array([distCoef, 0, 0, 0]))

    # Warp image to top-view
    cam_warped = cv2.warpPerspective(cam_undistort, H_cam2minimap, dsize=[(map.shape[1]), (map.shape[0])])
