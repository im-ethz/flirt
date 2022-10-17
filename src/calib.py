import torch

from src.utils import extract_pair
from src.utils import distort_point
from src.utils import match_minimap
from src.utils import ground_1camloss
from src.utils import make_loss_for_n
from src.utils import extract_1cam_map
from src.utils import get_intersection
from src.utils import normalize_points
from src.utils import optimize_parameters
from src.utils import homography_to_camera
from src.utils import get_homography_matrix
from src.utils import loss_essential_matrix
from src.utils import get_fundamental_matrix
from src.utils import matching_to_base_reach
from src.utils import camera_matrix_to_caminfo
from src.utils import essential_matrix_decomposition

def calib_2cam(normalized_point_info, normalized_map_point_info, cam0, cam1, floor_list, n_cam):
    """
    Calibrate 2 cameras together with a minimap
    """
    normalized_points0, normalized_points1, n_matching, pair_floor_list = \
        extract_pair(normalized_point_info, [cam0, cam1], floor_list)
    cam_matching0, map_matching0, _ = extract_1cam_map(normalized_map_point_info, cam0, n_cam)
    cam_matching1, map_matching1, _ = extract_1cam_map(normalized_map_point_info, cam1, n_cam)
    normalized_cam_point = {0: cam_matching0, 1: cam_matching1}
    map_matching = {0: map_matching0, 1: map_matching1}
    base_point, reach_point, base_minimap_points, reach_minimap_points = \
        matching_to_base_reach(map_matching, normalized_cam_point)

    one = torch.ones(1, dtype=torch.float32)
    best = 1e10
    for K0 in range(2):
        K0 = K0 * 0.4 * one
        for K1 in range(2):
            K1 = K1 * 0.4 * one
            for log_f0 in range(2):
                log_f0 = log_f0 * 0.7 * one
                for log_f1 in range(2):
                    log_f1 = log_f1 * 0.7 * one
                    f0 = torch.exp(log_f0)
                    f1 = torch.exp(log_f1)
                    # TODO: calls function to distort points but output variable name is confusing (should be 'distorted_cam_point'). Moreover should be undistorting points here not distorting them as the image points are already distorted
                    undistorted_points0 = distort_point(normalized_points0, K0) / f0
                    undistorted_points1 = distort_point(normalized_points1, K1) / f1
                    fundamental_matrix = get_fundamental_matrix(undistorted_points0, undistorted_points1)
                    parameters = torch.cat([fundamental_matrix.view(-1), K0, K1, log_f0, log_f1], dim=0).requires_grad_(
                        True)
                    loss_function = loss_essential_matrix(normalized_points0, normalized_points1)
                    parameters, loss = optimize_parameters(parameters, loss_function.loss, 10)
                    if loss < best:
                        best = loss
                        best_parameters = parameters

    essential_matrix, K0, K1, log_f0, log_f1 = torch.split(best_parameters, [9, 1, 1, 1, 1])
    f0 = torch.exp(log_f0)
    f1 = torch.exp(log_f1)
    essential_matrix = essential_matrix.view(3, 3)
    # TODO: calls function to distort points but output variable name is confusing (should be 'distorted_cam_point'). Moreover should be undistorting points here not distorting them as the image points are already distorted
    undistorted_points0 = distort_point(normalized_points0, K0) / f0
    undistorted_points1 = distort_point(normalized_points1, K1) / f1
    if base_point[0, 2] == 0:
        base_point[:, :2] = distort_point(base_point[:, :2], K0) / f0
    if base_point[0, 2] == 1:
        base_point[:, :2] = distort_point(base_point[:, :2], K1) / f1
    if reach_point[0, 2] == 0:
        reach_point[:, :2] = distort_point(reach_point[:, :2], K0) / f0
    if reach_point[0, 2] == 1:
        reach_point[:, :2] = distort_point(reach_point[:, :2], K1) / f1
    T, R = essential_matrix_decomposition(essential_matrix, undistorted_points0, undistorted_points1)
    cam0_to_cam1 = -torch.matmul(torch.inverse(R), T).squeeze(-1)

    n_points = undistorted_points0.shape[0]
    intersection, _, _, _ = get_intersection(undistorted_points0, undistorted_points1,
                                             torch.ones([n_points, 1], dtype=torch.float32),
                                             torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(n_points, 3, 3),
                                             torch.zeros([n_points, 3], dtype=torch.float32),
                                             torch.ones([n_points, 1], dtype=torch.float32),
                                             torch.inverse(R).unsqueeze(0).expand(n_points, 3, 3),
                                             cam0_to_cam1.unsqueeze(0).expand(n_points, 3))

    intersection_on_floor = intersection[pair_floor_list]

    thetas0, thetas1, t0, t1 = match_minimap(intersection_on_floor, base_point, reach_point, base_minimap_points,
                                             reach_minimap_points, R, T.unsqueeze(1), cam0_to_cam1)

    parameters0 = torch.cat([thetas0, t0, f0, K0], dim=0)
    parameters1 = torch.cat([thetas1, t1, f1, K1], dim=0)

    parameters_dict = {cam0:parameters0,cam1:parameters1}

    return parameters_dict

def calib_1cam(current_cam_id, calibrated_points, normalized_map_point_info, n_cam, normalized_saved):
    cam_matching, map_matching, map_matching_ids = extract_1cam_map(normalized_map_point_info,
                                                                    current_cam_id, n_cam)
    if len(set(calibrated_points.keys()) - set(map_matching_ids)) == 0:
        normalized_cam_point = cam_matching
        map_points = map_matching
    else:
        normalized_cam_point = []
        map_points = []
        for key in sorted(list(set(calibrated_points.keys()) - set(map_matching_ids))):
            normalized_cam_point.append(normalized_saved[current_cam_id][key])
            map_points.append(calibrated_points[key])
        normalized_cam_point = torch.stack(normalized_cam_point, dim=0)
        normalized_cam_point = torch.cat([normalized_cam_point, cam_matching], dim=0)
        map_points = torch.cat(map_points, dim=0)
        map_points = torch.cat([map_points, map_matching], dim=0)
    map_points, norm_factor_map, center_for_norm_map = normalize_points(map_points)

    one = torch.ones(1, dtype=torch.float32)
    best = 1e10
    for K in range(5):
        K = K * 0.1 * one
        # TODO: calls function to distort points but output variable name is confusing (should be 'distorted_cam_point'). Moreover should be undistorting points here not distorting them as the image points are already distorted
        undistorted_points = distort_point(normalized_cam_point, K)
        homography_matrix = get_homography_matrix(undistorted_points, map_points)
        camera_matrix = homography_to_camera(homography_matrix)
        parameters = camera_matrix.view(-1)
        parameters = torch.cat([parameters, K], dim=0)
        parameters = parameters.requires_grad_(True)
        loss_function = ground_1camloss(map_points, normalized_cam_point)

        parameters, loss = optimize_parameters(parameters, loss_function.loss, 10)

        if loss < best:
            best = loss
            best_parameters = parameters
    camera_matrix, K = torch.split(best_parameters, [12, 1])
    thetas, t, f = camera_matrix_to_caminfo(camera_matrix, norm_factor_map, center_for_norm_map)
    parameters = torch.cat([thetas, t, f, K], dim=0)
    return parameters

def calib_ncam(parameters_dict, normalized_saved, n_cam, normalized_map_point_info, normalized_point_info, floor_list):
    cam_list = sorted(list(parameters_dict.keys()))
    parameters = torch.cat([parameters_dict[key] for key in cam_list],dim=0).requires_grad_(True)
    loss_function = make_loss_for_n(normalized_saved, cam_list,
                                        n_cam,normalized_map_point_info,normalized_point_info,floor_list)
    parameters, loss = optimize_parameters(parameters, loss_function.loss, 50)
    for i in range(len(cam_list)):
        parameters_dict[cam_list[i]] = parameters[8 * i:8 * (i + 1)]
    return parameters_dict
