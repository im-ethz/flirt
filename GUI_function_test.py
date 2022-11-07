import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import saved_to_info, get_normalized_points_dict, get_precal_coords
from src.visualize import display_result, pointing_puttext
from src.calib import calib_2cam, calib_1cam, calib_ncam



cam_list = ['CAM21', 'CAM22', 'CAM23', 'CAM24', 'CAM25', 'CAM26', 'CAM27', 'CAM28', 'CAM29', 'CAM30', 'CAM31', 'CAM32']
# cam_list = ['1', '2', '3', '4', '5', '6', '7', '8']
floor_list = [0, 1, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
              33, 35, 36, 38, 39, 40, 41, 42,
              43, 44, 45, 47, 48, 49, 51, 52]
calibration_sequence = [1, 2, 0, 3, 4, 6, 5, 8, 7, 11, 10, 9]

# saved_path = '/home/jeewook/projects/calibrate_matrix/save_result1/' + saved_path + 'CoexClientMap_4x_pillar_0/cam.npy'
saved_path = './calibration_example_data/COEX/cam.npy'
saved = np.load(saved_path, allow_pickle=True).item()

n_cam = len(saved) - 1
cam = {}
for cam_index in range(n_cam):
    # cam[cam_index] = cv2.imread(f'/home/jeewook/data/coex/images/{cam_list[cam_index]}.png')
    cam[cam_index] = cv2.imread(f'./calibration_example_data/COEX/{cam_list[cam_index]}.png')

# minimap = cv2.imread('/home/jeewook/data/coex/CoexClientMap_4x_pillar.png')
minimap = cv2.imread('./calibration_example_data/COEX/CoexClientMap_4x_pillar.png')

saved2 = {}
for cam_key in saved.keys():
    saved2[int(cam_key)] = {}
    for point_key in saved[cam_key].keys():
        saved2[int(cam_key)][int(point_key)] = saved[cam_key][point_key]
saved = saved2

# point_info, map_point_info = saved_to_info(saved, n_cam)
normalized_saved = get_normalized_points_dict(saved, n_cam, cam)
normalized_point_info, normalized_map_point_info = saved_to_info(normalized_saved, n_cam)

parameters_dict = calib_2cam(normalized_point_info, normalized_map_point_info,
                                calibration_sequence[0], calibration_sequence[1],
                                floor_list, 
                                n_cam)

display_result(parameters_dict[calibration_sequence[0]], cam[calibration_sequence[0]], minimap, True)
display_result(parameters_dict[calibration_sequence[1]], cam[calibration_sequence[1]], minimap, True)

parameters_dict = calib_ncam(parameters_dict,normalized_saved,n_cam,normalized_map_point_info,normalized_point_info,floor_list)

display_result(parameters_dict[calibration_sequence[0]], cam[calibration_sequence[0]], minimap, True)
display_result(parameters_dict[calibration_sequence[1]], cam[calibration_sequence[1]], minimap, True)

for nth_cam in range(2, len(calibration_sequence)):
    calibrated_camids = calibration_sequence[:nth_cam]
    
    

    calibrated_points = get_precal_coords(calibration_sequence[nth_cam], calibrated_camids, parameters_dict,
                                          normalized_point_info,floor_list)
    pointed_map = minimap.copy()
    import pdb;pdb.set_trace()
    for point_id in calibrated_points.keys():
        pointed_map = pointing_puttext(pointed_map, point_id,
                                       tuple(calibrated_points[point_id][0].numpy().astype(np.int32)), (0, 0, 255))
    plt.figure()
    plt.imshow(cv2.cvtColor(pointed_map, cv2.COLOR_BGR2RGB))

    parameters = calib_1cam(calibration_sequence[nth_cam], calibrated_points, normalized_map_point_info, n_cam, normalized_saved)

    display_result(parameters, cam[calibration_sequence[nth_cam]], minimap, True)

    parameters_dict[calibration_sequence[nth_cam]] = parameters
    parameters_dict = calib_ncam(parameters_dict, normalized_saved, n_cam, normalized_map_point_info,
                                 normalized_point_info, floor_list)

    for key in parameters_dict.keys():
        display_result(parameters_dict[key], cam[key], minimap, True)



