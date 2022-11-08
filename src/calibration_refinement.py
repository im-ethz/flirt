import numpy as np
import cv2
import os
from typing import Tuple, List
import matplotlib.pyplot as plt
import argparse

from scipy.optimize import least_squares, minimize

def thetas_to_R(thetas,change_system):
    zero = np.zeros(1,dtype=np.float32)
    thetas_x, thetas_y, thetas_z = thetas[:,None]
    Rx = np.stack([np.concatenate([zero + 1, zero, zero], axis=0),
                      np.concatenate([zero, np.cos(thetas_x), -np.sin(thetas_x)], axis=0),
                      np.concatenate([zero, np.sin(thetas_x), np.cos(thetas_x)], axis=0)], axis=0)
    Ry = np.stack([np.concatenate([np.cos(thetas_y),zero,np.sin(thetas_y)], axis=0),
                      np.concatenate([zero,zero+1,zero], axis=0),
                      np.concatenate([-np.sin(thetas_y),zero,np.cos(thetas_y)], axis=0)], axis=0)
    Rz = np.stack([np.concatenate([np.cos(thetas_z),-np.sin(thetas_z),zero], axis=0),
                      np.concatenate([np.sin(thetas_z),np.cos(thetas_z),zero], axis=0),
                      np.concatenate([zero,zero,zero+1], axis=0)], axis=0)
    R = np.matmul(Rz,np.matmul(Ry,Rx))
    if change_system:
        R = R * np.array([1, 1, -1], dtype=np.float32)[None]
    return R

class Camera():
    def __init__(self, img, initial_guess):
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_shape = [self.image.shape[1], self.image.shape[0]]

        # Extract raw intrinsics and extrinsics from calibration parameters
        self.euler, self.C, self.f, self.k = np.split(initial_guess, [3, 6, 7])

    def update(self, camera_parameters):
        self.euler, self.C, self.f, self.k = np.split(camera_parameters, [3, 6, 7])
    
    def construct_H(self):
        self.R = thetas_to_R(self.euler, True)
        t = -self.R @ self.C.reshape(-1, 1)

        # Get normalization parameters from image size
        image_center = np.array(self.img_shape, dtype=np.float32) / 2
        image_norm_length = np.sqrt(np.mean(image_center ** 2))

        # Construct intrinsic matrix K
        self.K = np.array([[self.f[0]*image_norm_length, 0, image_center[0]], [0, self.f[0]*image_norm_length, image_center[1]], [0, 0, 1]], dtype=np.float32)

        # Construct 2D extrinsic matrix m (for homography)
        m = np.concatenate((self.R[:, :2], t), axis=1)

        # Construct 2D projection matrix, i.e. homography matrix H (3x3)
        H = self.K @ m

        return H

    def construct_H_inv(self):
        H = self.construct_H()
        H_inv = np.linalg.inv(H)
        return H_inv

    def img_to_minimap(self, minimap_shape):
        # get parameters
        H_inv = self.construct_H_inv()

        # Undistort image
        img_undistort = cv2.undistort(self.image, self.K, np.array([-self.k[0], 0, 0, 0]))

        # Warp image to top-view
        img_warped = cv2.warpPerspective(img_undistort, H_inv, dsize=minimap_shape, flags=cv2.INTER_LINEAR)

        return img_warped

    def minimap_to_img(self):
        pass

    def get_edge_img(self, edge_method = 'sobel'):
        # Convert to graycsale
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

        if edge_method == 'sobel':
            # Sobel Edge Detection
            edge_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        elif edge_method == 'canny':
            # Canny Edge Detection
            edge_img = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

        return edge_img

    def plot(self, minimap_shape):
        img_warped = self.img_to_minimap(minimap_shape)
        plt.figure(figsize=(15, 9))
        plt.imshow(np.uint(img_warped))


# Compute photometric error (i.e., the residuals)
def computeResiduals(camera_parameters: np.ndarray, cameras: List[Camera], minimap_shape: Tuple[int, int], mask: np.ndarray):
    """
    Computes the image alignment error (residuals). 
    """
    # update cameras with new camera parameters
    camera_parameters = camera_parameters.reshape(len(cameras), -1)
    for i, camera in enumerate(cameras):
        camera.update(camera_parameters[i])
    
    residuals = []
    # Warp each point in the previous image, to the current image
    for i, cam1 in enumerate(cameras):
        for j, cam2 in enumerate(cameras):
            if i >= j:
                continue

            # warp both images to minimap
            warped1 = cam1.img_to_minimap(minimap_shape)
            # mask1 = np.logical_and(warped1[:, :, 0]>0, (warped1[:, :, 1]>0))
            # mask1 = np.logical_and(mask1, warped1[:, :, 2]>0)
            warped2 = cam2.img_to_minimap(minimap_shape)
            # mask2 = np.logical_and(warped2[:, :, 0]>0, (warped2[:, :, 1]>0))
            # mask2 = np.logical_and(mask2, warped2[:, :, 2]>0)
            # mask12 = np.bool8(np.logical_and(mask1, mask2))
            # combined_mask = np.logical_and(mask, mask12)
            combined_mask = mask
            residuals.append((np.mean(warped1, axis=2)-np.mean(warped2, axis=2))*combined_mask)
    residuals = np.asarray(residuals)
    return residuals.flatten()

# Compute photometric error (i.e., the residuals)
def computeLoss(camera_parameters: np.ndarray, cameras: List[Camera], minimap_shape: Tuple[int, int], mask: np.ndarray):
    """
    Computes the image alignment loss function. 
    """
    residuals = computeResiduals(camera_parameters, cameras, minimap_shape, mask)

    loss = 0.5 * np.sum(np.square(residuals))

    return loss

def main(args):

    # Initialize camera classes for every camera
    cam_ids = args.cam_ids 
    assert len(cam_ids) > 1

    map = cv2.imread(args.minimap_path, cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.imshow(map)
    # plt.show()
    cameras = []
    initial_guess = []
    for cam_id in cam_ids:
        cam_path = os.path.join(args.images_folder, str(cam_id) + '.png')
        npy_file_path = os.path.join(args.parameters_folder, str(cam_id) + '.npy')
        img = cv2.resize(cv2.imread(cam_path), (640, 480))
        cameras_parameters = np.load(npy_file_path)
        initial_guess.append(cameras_parameters)
        cameras.append(Camera(img, cameras_parameters))
        cameras[-1].plot((map.shape[1], map.shape[0]))

    # Get area of minimap to compute loss (mask for loss)
    mask = np.zeros_like(map, dtype=np.bool8)
    mask[172:445, 292:1108] = True
    
    # Perform least squares optimization
    initial_guess = np.asarray(initial_guess).reshape(-1,)
    loss = 0.5 * np.sum(np.square(computeResiduals(initial_guess, cameras, (map.shape[1], map.shape[0]), mask)))
    print("loss before refinement: ", loss)
    # res = least_squares(computeResiduals, initial_guess, args=(cameras, (map.shape[1], map.shape[0]), mask), jac='3-point', method='trf', verbose=2, ftol=1e-09, xtol=1e-10, gtol=1e-09, loss='linear')
    res = minimize(computeLoss, initial_guess, args=(cameras, (map.shape[1], map.shape[0]), mask), method='Nelder-Mead', options={'maxiter': 10000, 'disp':True})
    loss = 0.5 * np.sum(np.square(computeResiduals(res.x, cameras, (map.shape[1], map.shape[0]), mask)))
    print("loss after refinement: ", loss)

    print("camera parameters diff", np.abs(initial_guess - res.x))

    print("new params", res.x)
    for cam in cameras:
        cam.plot((map.shape[1], map.shape[0]))
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calibration Refinement")
    # Data path parameters
    parser.add_argument("--minimap-path", required=True, type=str,
                        help="Path to the minimap image")
    parser.add_argument("--images-folder", required=True, type=str,
                        help="Path to images folder")
    parser.add_argument("--parameters-folder", required=True, type=str,
                        help="Path to camera parameters")
    parser.add_argument("--cam-ids", required=True, nargs="+",
                        type=int, default=[])

    args = parser.parse_args()

    main(args)
    
    


