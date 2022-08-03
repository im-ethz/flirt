import os
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt

from src.utils import normalize_points_by_image, cam_to_plane

def pointing_puttext(pointed_cam, point_id, point_coords, color):
    """점들을 실제로 찍는 함수"""
    half_diag_length = pointed_cam.shape
    half_diag_length = np.sqrt(half_diag_length[0] ** 2 + half_diag_length[1] ** 2) / 2
    pointed_cam = cv2.circle(pointed_cam, point_coords, int(half_diag_length / 500) + 1, color,
                                             thickness=int(half_diag_length / 500) + 2)
    pointed_cam = cv2.putText(pointed_cam, str(point_id),(point_coords[0], point_coords[1] - 5), fontFace=0,
        fontScale=half_diag_length / 1000, color=color, thickness=int(half_diag_length / 2000) + 1)
    return pointed_cam

def visualize_input(cam, minimap, point_info, map_point_info, n_cam):
    """들어온 점 정보들을 모든 카메라 이미지와 minimap에 표시하여 띄운다"""
    pointed_cam = {}
    for cam_index in range(n_cam):
        pointed_cam[cam_index] = cam[cam_index].copy()
    for point_id in point_info.keys():
        for cam_index in point_info[point_id].keys():
            pointed_cam[cam_index] = pointing_puttext(pointed_cam[cam_index],
                                                    point_id, point_info[point_id][cam_index], (255, 0, 0))
    pointed_map = minimap.copy()
    for point_id in map_point_info.keys():
        for cam_index in map_point_info[point_id].keys():
            if cam_index >= n_cam:
                continue
            pointed_cam[cam_index] = pointing_puttext(pointed_cam[cam_index],
                                                           point_id, map_point_info[point_id][cam_index], (0, 0, 255))
        pointed_map = pointing_puttext(pointed_map, point_id, map_point_info[point_id][n_cam], (0, 0, 255))
    for cam_index in range(n_cam):
        plt.figure()
        plt.imshow(cv2.cvtColor(pointed_cam[cam_index], cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(pointed_map, cv2.COLOR_BGR2RGB))

def display_result(parameters, cam, plane_figure, save_data=False, save_path = './temp_image_save/'):
    """calibrate 된 카메라 화면에 등간격으로 점을 찍고, 이 점들을 평면도에 mapping한 결과를 띄우거나 저장하는 함수"""
    pointed_cam = cam.copy()
    pointed_plane_figure = plane_figure.copy()
    half_diag_length = pointed_cam.shape
    half_diag_length = np.sqrt(half_diag_length[0] ** 2 + half_diag_length[1] ** 2) / 2
    half_diag_length_plane = pointed_plane_figure.shape
    half_diag_length_plane = np.sqrt(half_diag_length_plane[0] ** 2 + half_diag_length_plane[1] ** 2) / 2

    coords_dist = half_diag_length/7
    n_xpoints = int(cam.shape[1] // coords_dist + 1)
    n_ypoints = int(cam.shape[0] // coords_dist + 1)
    coords_x = coords_dist * torch.arange(n_xpoints).unsqueeze(1).expand(n_xpoints, n_ypoints)
    coords_y = coords_dist * torch.arange(n_ypoints).unsqueeze(0).expand(n_xpoints, n_ypoints)
    cam_point = torch.stack([coords_x, coords_y], dim=2).reshape(-1,2)
    normalized_cam_point = normalize_points_by_image(cam_point, cam.shape[1::-1])
    ground_point = cam_to_plane(normalized_cam_point, parameters)

    for i in range(cam_point.shape[0]):
        pointed_cam = cv2.circle(pointed_cam, (int(cam_point[i][0]), int(cam_point[i][1])), int(half_diag_length/2000)+1,
                                  (255, 0, 0), thickness=int(half_diag_length/2000)+2)
        pointed_plane_figure = cv2.circle(pointed_plane_figure, (int(ground_point[i][0]), int(ground_point[i][1])),
                int(half_diag_length_plane/2000)+1, (255, 0, 0), thickness=int(half_diag_length_plane/2000)+2)
    if save_data:
        os.makedirs(save_path, exist_ok=True)
        i=0
        while True:
            if os.path.isfile(save_path+str(i)+'.jpg'):
                i+=1
            else:
                break
        plt.imsave(save_path+str(i)+'.jpg', cv2.cvtColor(pointed_cam, cv2.COLOR_BGR2RGB))
        i=0
        while True:
            if os.path.isfile(save_path+str(i)+'.jpg'):
                i+=1
            else:
                break
        plt.imsave(save_path+str(i)+'.jpg', cv2.cvtColor(pointed_plane_figure, cv2.COLOR_BGR2RGB))
    else:
        plt.figure()
        plt.imshow(cv2.cvtColor(pointed_cam, cv2.COLOR_BGR2RGB))
        plt.figure()
        plt.imshow(cv2.cvtColor(pointed_plane_figure, cv2.COLOR_BGR2RGB))
