import copy
import torch

import numpy as np
import torch.nn.functional as F


def get_intersection(points0, points1, f0, R_inv0, t0,f1,R_inv1, t1):
    """카메라의 점, 카메라 파라미터가 서로 다른 카메라에 있는 2개 점에 대해 주어졌을 때 교점을 구하는 함수"""
    vector0 = points0 / f0
    vector0 = torch.matmul(R_inv0, F.pad(vector0, [0, 1], value=1).unsqueeze(2)).squeeze(2)
    vector0 = vector0 / torch.sqrt(torch.sum(vector0 ** 2, dim=1, keepdim=True))
    vector1 = points1 / f1
    vector1 = torch.matmul(R_inv1, F.pad(vector1, [0, 1], value=1).unsqueeze(2)).squeeze(2)
    vector1 = vector1 / torch.sqrt(torch.sum(vector1 ** 2, dim=1, keepdim=True))
    v0dotv1 = torch.sum(vector0 * vector1, dim=1, keepdim=True)
    t0substractt1 = t0 - t1
    distance0 = (torch.sum(t0substractt1 * vector1, dim=1, keepdim=True) * v0dotv1 - \
                 torch.sum(t0substractt1 * vector0, dim=1, keepdim=True)) / (1 - v0dotv1 ** 2)
    distance1 = (torch.sum(t0substractt1 * vector1, dim=1, keepdim=True) - \
                 torch.sum(t0substractt1 * vector0, dim=1, keepdim=True) * v0dotv1) / (1 - v0dotv1 ** 2)
    intersection0 = t0 + distance0 * vector0
    intersection1 = t1 + distance1 * vector1
    intersection = (intersection0 * distance1 ** 2 + intersection1 * distance0 ** 2) / (distance0 ** 2 + distance1 ** 2)
    errors = torch.sum((intersection0-intersection1)**2)
    return intersection, distance0, distance1, errors

def get_calibrated_points(parameters, cam_shape, num_points=10, offset=10):
    """calibrate 된 카메라 화면에 등간격으로 점을 찍고, 이 점들을 평면도에 mapping한 결과를 띄우거나 저장하는 함수"""
    spacing_x = (cam_shape[1] - (2 * offset)) / (num_points - 1)
    spacing_y = (cam_shape[0] - (2 * offset)) / (num_points - 1)
    
    coords_x = spacing_x * torch.arange(num_points).unsqueeze(1).expand(num_points, num_points) + offset
    coords_y = spacing_y * torch.arange(num_points).unsqueeze(0).expand(num_points, num_points) + offset
    
    cam_point = torch.stack([coords_x, coords_y], dim=2).reshape(-1,2)
    normalized_cam_point = normalize_points_by_image(cam_point, cam_shape[1::-1])
    ground_point = cam_to_plane(normalized_cam_point, parameters)

    return cam_point, ground_point

def cam_to_plane(normalized_cam_point, parameters):
    thetas, t, f, K = torch.split(parameters, [3, 3, 1, 1])
    R = thetas_to_R(thetas.unsqueeze(0), True).squeeze(0)
    undistorted_cam_point = distort_point(normalized_cam_point, K)
    vector0 = undistorted_cam_point / f
    vector0 = torch.matmul(torch.inverse(R).unsqueeze(0),
                           F.pad(vector0, [0, 1], value=1).unsqueeze(2)).squeeze(2)
    vector0 = vector0 / torch.sqrt(torch.sum(vector0 ** 2, dim=1, keepdim=True))
    ground_point = t[None] - (t[None, 2:] / vector0[:, 2:]) * vector0
    return ground_point[:,:2]


def saved_normalize(saved, n_cam, cam):
    """saved에 저장된 값들은 픽셀 좌표라 값들이 100~1000scale이며 모두 양수이다. 중심을 이미지의 중심으로 맞춰주고,
     scale을 1 근처로 맞춰준다."""
    normalized_saved = copy.deepcopy(saved)
    for camid in range(n_cam):
        points = []
        for pointid in normalized_saved[camid].keys():
            points.append(torch.tensor(normalized_saved[camid][pointid], dtype=torch.float32))
        points = torch.stack(points, dim=0)
        normalized_points = normalize_points_by_image(points, cam[camid][1::-1])
        index = 0
        for pointid in normalized_saved[camid].keys():
            normalized_saved[camid][pointid] = normalized_points[index]
            index += 1
    return normalized_saved

def normalize_points_by_image(points, cam_shape_xy):
    image_center = torch.tensor(cam_shape_xy, dtype=torch.float32) / 2
    image_norm_length = torch.sqrt(torch.mean(image_center ** 2))
    normalized_points = (points - image_center[None]) / image_norm_length
    return normalized_points

def optimize_parameters(parameters, loss_function, num_iter):
    """뉴턴법으로 local_minima 찾는 함수
    이 함수가 시간이 가장 오래 걸림"""
    best_loss = loss_function(parameters)
    best_parameters = parameters
    prev_best_loss = 1e21
    loss = 1e21
    i = 0
    while prev_best_loss != best_loss and i!=num_iter:
        prev_best_loss = best_loss
        rp = loss_function(parameters)
        # if i == 0:
            # print(rp.item())
        rp.backward()
        parameters_grad = parameters.grad
        H = torch.autograd.functional.hessian(loss_function, parameters, vectorize=True)
        mu = 10 ** (-6)
        while mu <= 10 ** 3:
            try:
                Hinv = torch.inverse(H + mu * torch.diag_embed(torch.diagonal(H)))
            except:
                mu *= 10
                continue
            parametersstar = parameters - torch.matmul(Hinv, parameters_grad.unsqueeze(1)).squeeze(1)
            try:
                loss = loss_function(parametersstar)
            except:
                mu *= 10
                continue
            if best_loss > loss:
                best_mu = mu
                best_parameters = parametersstar
                best_loss = loss
                break
            mu *= 10
        parameters = best_parameters.clone().detach().requires_grad_(True)
        # print(i, best_loss.item(), best_mu)
        i += 1
    return best_parameters.detach(), best_loss

def distort_point(point, distortion_parameters):
    """렌즈왜곡 펴는 함수"""
    x = point[:,0:1]
    y = point[:,1:2]
    r_square = x**2 + y**2
    point = point * (1 + distortion_parameters * r_square)
    return point

def R_to_thetas(R, change_system):
    if change_system:
        R = R * torch.tensor([1, 1, -1], dtype=torch.float32).unsqueeze(0)
    thetax = torch.atan2(R[2,1],R[2,2])
    thetay = torch.atan2(-R[2, 0], torch.sqrt(R[2, 1]**2+R[2, 2]**2))
    thetaz = torch.atan2(R[1, 0], R[0, 0])
    thetas = torch.stack([thetax,thetay,thetaz],dim=0)
    return thetas

def thetas_to_R(thetas, change_system):
    zero = torch.zeros(thetas.shape[0],dtype=torch.float32,device=thetas.device)
    thetas_x, thetas_y, thetas_z = thetas[:,0], thetas[:,1], thetas[:,2]
    Rx = torch.stack([torch.stack([zero + 1, zero, zero], dim=1),
                      torch.stack([zero, torch.cos(thetas_x), -torch.sin(thetas_x)], dim=1),
                      torch.stack([zero, torch.sin(thetas_x), torch.cos(thetas_x)], dim=1)], dim=1)
    Ry = torch.stack([torch.stack([torch.cos(thetas_y),zero,torch.sin(thetas_y)], dim=1),
                      torch.stack([zero,zero+1,zero], dim=1),
                      torch.stack([-torch.sin(thetas_y),zero,torch.cos(thetas_y)], dim=1)], dim=1)
    Rz = torch.stack([torch.stack([torch.cos(thetas_z),-torch.sin(thetas_z),zero], dim=1),
                      torch.stack([torch.sin(thetas_z),torch.cos(thetas_z),zero], dim=1),
                      torch.stack([zero,zero,zero+1], dim=1)], dim=1)
    R = torch.matmul(Rz,torch.matmul(Ry,Rx))
    if change_system:
        R = R * torch.tensor([1, 1, -1], dtype=torch.float32,device=thetas.device).unsqueeze(0).unsqueeze(0)
    return R

def homography_equation(points0, points1):
    x = points0[:, :1]
    y = points0[:, 1:]
    x, y, z = x / torch.sqrt(x ** 2 + y ** 2 + 1), y / torch.sqrt(x ** 2 + y ** 2 + 1), 1 / torch.sqrt(
        x ** 2 + y ** 2 + 1)
    x_prime = points1[:, :1]
    y_prime = points1[:, 1:]
    x_prime, y_prime, z_prime = x_prime / torch.sqrt(x_prime ** 2 + y_prime ** 2 + 1), \
                                y_prime / torch.sqrt(x_prime ** 2 + y_prime ** 2 + 1), \
                                1 / torch.sqrt(x_prime ** 2 + y_prime ** 2 + 1)
    homography_equation_matrix0 = torch.cat(
        [-x * z_prime, -y * z_prime, -z * z_prime, x * y_prime, y * y_prime, z * y_prime], dim=1)
    homography_equation_matrix0 = F.pad(homography_equation_matrix0, [3, 0])
    homography_equation_matrix1 = torch.cat(
        [-x * x_prime, -y * x_prime, -z * x_prime, x * z_prime, y * z_prime, z * z_prime], dim=1)
    homography_equation_matrix1 = F.pad(homography_equation_matrix1, [3, 0])
    homography_equation_matrix1 = torch.roll(homography_equation_matrix1, 3, dims=1)
    homography_equation_matrix = torch.cat([homography_equation_matrix0, homography_equation_matrix1], dim=0)
    if homography_equation_matrix.shape[0] == 8:
        homography_equation_matrix = F.pad(homography_equation_matrix, [0, 0, 0, 1])
    return homography_equation_matrix

def fundamental_equation(points0, points1):
    x = points0[:, :1]
    y = points0[:, 1:]
    x,y,z = x/torch.sqrt(x**2+y**2+1),y / torch.sqrt(x ** 2 + y ** 2 + 1),1 / torch.sqrt(x ** 2 + y ** 2 + 1)
    x_prime = points1[:, :1]
    y_prime = points1[:, 1:]
    x_prime,y_prime,z_prime = x_prime/torch.sqrt(x_prime**2+y_prime**2+1),\
        y_prime / torch.sqrt(x_prime ** 2 + y_prime ** 2 + 1),1 / torch.sqrt(x_prime ** 2 + y_prime ** 2 + 1)
    eight_point_matrix = torch.cat([x_prime * x, x_prime * y, x_prime * z, y_prime * x,
                        y_prime * y, y_prime * z, x * z_prime, y * z_prime, z * z_prime], dim=1)
    return eight_point_matrix

def homography_to_camera(homography_matrix):
    """homography matrix에서 camera matirx를 구한다."""
    a0sq = torch.sum(homography_matrix[0, :2] ** 2, dim=0, keepdim=True)
    a1sq = torch.sum(homography_matrix[1, :2] ** 2, dim=0, keepdim=True)
    a0dota1 = torch.sum(homography_matrix[0, :2] * homography_matrix[1, :2], dim=0, keepdim=True)
    temp = torch.sqrt(a0sq ** 2 + a1sq ** 2 - 2 * a0sq * a1sq + 4 * a0dota1 ** 2)
    x0 = torch.sqrt((a1sq - a0sq + temp) / 2)
    x1 = torch.sqrt((a0sq - a1sq + temp) / 2)
    if a0dota1 > 0:
        x1 = -x1
    x2 = - (x0 * torch.sum(homography_matrix[0, :2] * homography_matrix[2, :2]) +
            x1 * torch.sum(homography_matrix[1, :2] * homography_matrix[2, :2])) / (x0 ** 2 + x1 ** 2)
    x = torch.cat([x0, x1, x2], dim=0)
    camera_matrix = torch.cat([homography_matrix[:, :2], x.unsqueeze(1), homography_matrix[:, 2:]], dim=1)
    return camera_matrix

def saved_to_info(saved, n_cam):
    """cam_id 기준으로 point_id들이 저장되어 있는 것이 saved였다면 point_id 기준으로 cam_id들을 저장한 것이 point_info이고,
    특히 minimap과 대응된 point_id의 경우 map_point_info에 저장된다."""
    map_point_info = {}
    for point_id in saved[n_cam].keys():
        for cam_index in range(n_cam + 1):
            if point_id in saved[cam_index].keys():
                if not point_id in map_point_info.keys():
                    map_point_info[point_id] = {}
                map_point_info[point_id][cam_index] = saved[cam_index][point_id]
    point_id = 0
    point_info = {}
    while True:
        for cam_index in range(n_cam):
            if point_id in saved[cam_index].keys():
                if not point_id in point_info.keys():
                    point_info[point_id] = {}
                point_info[point_id][cam_index] = saved[cam_index][point_id]
        if not point_id in point_info.keys():
            break
        if len(point_info[point_id]) == 1:
            del (point_info[point_id])
        point_id += 1
    return point_info, map_point_info

def extract_pair(point_info, cam_id, floor_list):
    """딕셔너리로 된 point_info를 tensor로 바꿔준다."""
    matching0 = []
    matching1 = []
    pair_floor_list = []
    for point_id in point_info.keys():
        if cam_id[0] in point_info[point_id].keys() and cam_id[1] in point_info[point_id].keys():
            if point_id in floor_list:
                pair_floor_list.append(len(matching0))
            matching0.append(point_info[point_id][cam_id[0]])
            matching1.append(point_info[point_id][cam_id[1]])
    if matching0==[]:
        matching0 = torch.zeros((0,2),dtype=torch.float32)
        matching1 = torch.zeros((0, 2), dtype=torch.float32)
    else:
        matching0 = torch.stack(matching0,dim=0)
        matching1 = torch.stack(matching1, dim=0)
    n_matching = matching0.shape[0]
    return matching0, matching1, n_matching, pair_floor_list

def extract_1cam_map(map_point_info, camid, n_cam):
    """extract_pair와 비슷한데 map 대응점 처리하는 함수"""
    cam_matching = []
    map_matching = []
    map_matching_ids = []
    for point_id in map_point_info.keys():
        if camid in map_point_info[point_id].keys():
            map_matching_ids.append(point_id)
            cam_matching.append(map_point_info[point_id][camid])
            map_matching.append(torch.tensor(map_point_info[point_id][n_cam], dtype=torch.float32))
    if cam_matching == []:
        cam_matching = torch.zeros((0, 2), dtype=torch.float32)
        map_matching = torch.zeros((0, 2), dtype=torch.float32)
    else:
        cam_matching = torch.stack(cam_matching, dim=0)
        map_matching = torch.stack(map_matching, dim=0)
    return cam_matching, map_matching, map_matching_ids

def matching_to_base_reach(map_matching, normalized_cam_point):
    """두 카메라에 의해 구성된 가상의 3D 공간을 평면도로 matching 시키기 위한 두 점을 골라주는 함수"""
    torch_map_matching = torch.cat([map_matching[i] for i in range(len(map_matching))], dim=0)
    torch_idx_to_set_idx = {}
    index = 0
    point_id = 0
    for i in range(torch_map_matching.shape[0]):
        if point_id >= map_matching[index].shape[0]:
            point_id = 0
            index += 1
        torch_idx_to_set_idx[i] = [index, point_id]
        point_id += 1
    max_distance_index = torch.argmax(
        torch.sum((torch_map_matching.unsqueeze(1) - torch_map_matching.unsqueeze(0)) ** 2, dim=2)
            .reshape(-1)).item()
    base_point_id = max_distance_index // torch_map_matching.shape[0]
    base_point_id = torch_idx_to_set_idx[base_point_id]
    base_point = normalized_cam_point[base_point_id[0]][base_point_id[1]:base_point_id[1] + 1]
    base_point = F.pad(base_point, [0, 1], value=base_point_id[0])
    reach_point_id = max_distance_index % torch_map_matching.shape[0]
    reach_point_id = torch_idx_to_set_idx[reach_point_id]
    reach_point = normalized_cam_point[reach_point_id[0]][reach_point_id[1]:reach_point_id[1] + 1]
    reach_point = F.pad(reach_point, [0, 1], value=reach_point_id[0])
    base_minimap_points = map_matching[base_point_id[0]][base_point_id[1]]
    reach_minimap_points = map_matching[reach_point_id[0]][reach_point_id[1]]
    return base_point, reach_point, base_minimap_points, reach_minimap_points

def normalize_points(points):
    center = torch.mean(points, dim=0,keepdim=True)
    points = points - center
    mean_radius = torch.sqrt(torch.mean(torch.sum(points ** 2, dim=1),dim=0,keepdim=True)/2)
    points = points / mean_radius
    return points,mean_radius,center

def get_fundamental_matrix(normalized_points0, normalized_points1):
    """normalized 8_point algorithm으로 fundamental_matrix를 구한다."""
    zero = torch.zeros(1, dtype=torch.float32,device=normalized_points0.device)
    normalizedforF_points0,mean_length0,center_fornorm0 = normalize_points(normalized_points0)
    normalizedforF_points1,mean_length1,center_fornorm1 = normalize_points(normalized_points1)
    eight_point_matrix = fundamental_equation(normalizedforF_points0, normalizedforF_points1)
    _, s, v = torch.svd(eight_point_matrix)
    T0 = torch.stack([torch.cat([1 / mean_length0, zero, -center_fornorm0[:, 0] / mean_length0],dim=0),
                       torch.cat([zero, 1 / mean_length0, -center_fornorm0[:, 1] / mean_length0],dim=0),
                       torch.cat([zero, zero, zero + 1],dim=0)],dim=0)
    T1 = torch.stack([torch.cat([1 / mean_length1, zero, zero],dim=0), torch.cat([zero, 1 / mean_length1, zero],dim=0),
    torch.cat([-center_fornorm1[0, 0] / mean_length1, -center_fornorm1[0, 1] / mean_length1, zero + 1],dim=0)],dim=0)

    fundamental_matrix = torch.matmul(T1, torch.matmul(v[:, -1].view(3, 3), T0))
    return fundamental_matrix

def get_homography_matrix(normalized_cam_point, map_points):
    """normalized 8_point algorithm과 비슷한 방법으로 homography_matrix를 구한다."""
    cam_normalizedforF_points,cam_mean_length,cam_center_fornorm = normalize_points(normalized_cam_point)
    map_normalizedforF_points,map_mean_length,map_center_fornorm = normalize_points(map_points)

    homography_equation_matrix = homography_equation(map_normalizedforF_points, cam_normalizedforF_points)
    _, s, v = torch.svd(homography_equation_matrix)
    zero = torch.zeros(1, dtype=torch.float32)
    T0 = torch.stack([torch.cat([1 / map_mean_length, zero, -map_center_fornorm[:, 0] / map_mean_length], dim=0),
                      torch.cat([zero, 1 / map_mean_length, -map_center_fornorm[:, 1] / map_mean_length], dim=0),
                      torch.cat([zero, zero, zero + 1], dim=0)], dim=0)
    T1 = torch.stack([torch.cat([cam_mean_length, zero, cam_center_fornorm[:, 0]], dim=0),
                      torch.cat([zero, cam_mean_length, cam_center_fornorm[:, 1]], dim=0),
                      torch.cat([zero, zero, zero + 1], dim=0)], dim=0)
    homography_matrix = torch.matmul(T1, torch.matmul(v[:, -1].view(3, 3), T0))
    return homography_matrix

def normalize_points(points):
    """주어진 점들의 중심을 (0,0)으로 하고 중심으로부터의 거리를 루트(2)가 되게 normalize"""
    center = torch.mean(points, dim=0,keepdim=True)
    points = points - center
    mean_radius = torch.sqrt(torch.mean(torch.sum(points ** 2, dim=1),dim=0,keepdim=True)/2)
    points = points / mean_radius
    return points,mean_radius,center

class loss_essential_matrix():
    """essential matrix constraint, point correspondence constraint를 가장 잘 만족시키는 E, K,f를 찾는 함수"""
    def __init__(self, normalized_points0, normalized_points1):
        self.normalized_points0 = normalized_points0
        self.normalized_points1 = normalized_points1
    def loss(self, parameters):
        fundamental_matrix, K0, K1, log_f0, log_f1 = torch.split(parameters, [9, 1, 1, 1, 1])
        fundamental_matrix = fundamental_matrix.view(3, 3)
        f0 = torch.exp(log_f0)
        f1 = torch.exp(log_f1)
        undistorted_points0 = distort_point(self.normalized_points0, K0) / f0
        undistorted_points1 = distort_point(self.normalized_points1, K1) / f1

        normalizing_factor0 = normalize_points(undistorted_points0)[1]
        normalizing_factor1 = normalize_points(undistorted_points1)[1]

        u, s, v = torch.svd(fundamental_matrix)

        loss_E = (s[0] / s[1] - 1) ** 2 + (s[2] / s[1]) ** 2

        undistorted_points0 = F.pad(undistorted_points0, [0, 1], value=1)
        undistorted_points1 = F.pad(undistorted_points1, [0, 1], value=1)
        p1TF = torch.matmul(undistorted_points1.unsqueeze(1), fundamental_matrix.unsqueeze(0)).squeeze(1)
        Fp0 = torch.matmul(fundamental_matrix.unsqueeze(0), undistorted_points0.unsqueeze(2)).squeeze(2)
        p1TFp0 = torch.sum(undistorted_points1 * Fp0, dim=1, keepdim=True)
        loss = (p1TFp0) ** 2 / (normalizing_factor0 ** 2 * torch.sum(p1TF[:, :2] ** 2, dim=1, keepdim=True) +
                                normalizing_factor1 ** 2 * torch.sum(Fp0[:, :2] ** 2, dim=1, keepdim=True))
        loss = torch.sum(loss) + loss_E
        return loss

def essential_matrix_decomposition(E, normalized_points0, normalized_points1):
    """E에서 T,R을 찾는 함수"""
    n_points = normalized_points0.shape[0]
    best_n_pos_depth = 0
    u, s, v = torch.svd(E)
    for ambiguity1 in range(2):
        for ambiguity2 in range(2):
            for ambiguity3 in range(2):
                T = torch.matmul(u, torch.matmul(torch.tensor([[0, (-1) ** ambiguity1, 0],
                                                               [(-1) ** (ambiguity1 + 1), 0, 0], [0, 0, 0]],
                                                              dtype=torch.float32), u.permute(1, 0)))
                R = torch.matmul(u, torch.matmul(torch.tensor([[0, (-1) ** ambiguity2, 0],
                                                               [(-1) ** (ambiguity2 + 1), 0, 0],
                                                               [0, 0, (-1) ** ambiguity3]],
                                                              dtype=torch.float32), v.permute(1, 0)))
                R_right = thetas_to_R(R_to_thetas(R, False).unsqueeze(0), False).squeeze(0)
                if torch.sum((R-R_right)**2)>1e-3:
                    continue
                R_inv = torch.inverse(R)
                T = torch.tensor([T[2, 1], T[0, 2], T[1, 0]], dtype=torch.float32)
                cam0_to_cam1 = -torch.matmul(R_inv, T[..., None]).squeeze(-1)

                _, depth_cam0, depth_cam1, errors = get_intersection(normalized_points0, normalized_points1,
                    torch.ones([n_points,1],dtype=torch.float32),
                    torch.eye(3,dtype=torch.float32).unsqueeze(0).expand(n_points, 3, 3),
                    torch.zeros([n_points,3],dtype=torch.float32),torch.ones([n_points,1],dtype=torch.float32),
                    R_inv.unsqueeze(0).expand(n_points, 3, 3),cam0_to_cam1.unsqueeze(0).expand(n_points, 3))

                n_pos_depth = torch.sum(torch.where(depth_cam1 > 0, 1, 0)) + torch.sum(
                    torch.where(depth_cam0 > 0, 1, 0))
                if (n_pos_depth > best_n_pos_depth):
                    best_n_pos_depth = n_pos_depth
                    true_T = T
                    true_R = R
    T = true_T
    R = true_R
    return T, R

def match_minimap(intersection_on_floor, base_point, reach_point, base_minimap_points,
                  reach_minimap_points, R, T, cam0_to_cam1):
    """두 카메라에서 생성된 가상의 3D공간을 평면도와 matching한다."""
    intersection_on_floor_save = intersection_on_floor
    intersection_on_floor = intersection_on_floor - torch.mean(intersection_on_floor, dim=0, keepdim=True)
    normal_vector_floor = torch.svd(intersection_on_floor)[2][:, 2]
    distance_to_floor = torch.mean(torch.sum(intersection_on_floor_save * normal_vector_floor[None], dim=1))
    if distance_to_floor > 0:
        normal_vector_floor = -normal_vector_floor
    else:
        distance_to_floor = -distance_to_floor

    base_point_world = cam_to_plane_2cam(base_point, distance_to_floor, normal_vector_floor, R, cam0_to_cam1)
    reach_point_world = cam_to_plane_2cam(reach_point, distance_to_floor, normal_vector_floor, R, cam0_to_cam1)
    base_to_reach = (reach_point_world - base_point_world)[0]
    to_minimapgrid_scale = torch.sqrt(((reach_minimap_points[0] - base_minimap_points[0]) ** 2 +
                                       (reach_minimap_points[1] - base_minimap_points[1]) ** 2) / torch.sum(
        base_to_reach ** 2))
    base_to_reach_perpendicular = cross_product(base_to_reach[None], normal_vector_floor[None])[0]
    x_vector = (reach_minimap_points[0] - base_minimap_points[0]) * base_to_reach - \
               (reach_minimap_points[1] - base_minimap_points[1]) * base_to_reach_perpendicular
    x_vector = x_vector / torch.sqrt(torch.sum(x_vector ** 2))
    y_vector = cross_product(x_vector[None], normal_vector_floor[None])[0]

    R0 = torch.stack([x_vector, y_vector, normal_vector_floor], dim=1)
    t0 = F.pad(base_minimap_points, [0, 1]) - \
         torch.matmul(torch.inverse(R0), base_point_world.permute(1, 0)).squeeze(1) * to_minimapgrid_scale

    R1 = torch.matmul(R, R0)
    t1 = t0 - to_minimapgrid_scale * torch.matmul(torch.inverse(R1), T).squeeze(1)

    thetas0 = R_to_thetas(R0, True)
    thetas1 = R_to_thetas(R1, True)
    return thetas0, thetas1, t0, t1

def cam_to_plane_2cam(points, distance_to_floor, normal_vector_floor, R, cam0_to_cam1):
    points_world =[]
    for i in range(points.shape[0]):
        if points[i:i+1, 2] == 0:
            point_world = points[i:i+1, :2]
            point_world = F.pad(point_world, [0, 1], value=1)
            point_world = -distance_to_floor / torch.sum(point_world * normal_vector_floor[None],
                                                             dim=1, keepdim=True) * point_world
        elif points[i:i+1, 2] == 1:
            point_world = points[i:i+1, :2]
            point_world = F.pad(point_world, [0, 1], value=1)
            point_world = torch.matmul(torch.inverse(R)[None], point_world[..., None]).squeeze(-1)
            point_world = (torch.sum(-cam0_to_cam1[None] * normal_vector_floor) - distance_to_floor) / \
                              torch.sum(point_world * normal_vector_floor) * point_world + cam0_to_cam1[None]
        points_world.append(point_world)
    points_world = torch.cat(points_world,dim=0)
    return points_world

def cross_product(a, b):
    return torch.stack([a[:,1]*b[:,2]-a[:,2]*b[:,1],
                                 a[:,2]*b[:,0]-a[:,0]*b[:,2],
                                 a[:,0]*b[:,1]-a[:,1]*b[:,0]],dim=1)

def get_precal_coords(current_cam_id, calibrated_camids, parameters_dict, normalized_point_info, floor_list):
    """이미 calibrate 된 카메라의 바닥 점들의 평면도 대응점의 좌표를 구하는 함수"""
    calibrated_camids = set(calibrated_camids)
    calibrated_points = {}
    for point_id in normalized_point_info.keys():
        if current_cam_id in normalized_point_info[point_id].keys() and \
                point_id in floor_list:
            pre_cal_camlist = list(set(normalized_point_info[point_id].keys()).intersection(calibrated_camids))[:2]
            if len(pre_cal_camlist) == 0:
                continue
            if len(pre_cal_camlist) == 1:
                calibrated_points[point_id] = cam_to_plane(
                    normalized_point_info[point_id][pre_cal_camlist[0]].unsqueeze(0),
                    parameters_dict[pre_cal_camlist[0]])
            if len(pre_cal_camlist) == 2:
                point0 = distort_point(normalized_point_info[point_id][pre_cal_camlist[0]].unsqueeze(0),
                                       parameters_dict[pre_cal_camlist[0]][7])
                point1 = distort_point(normalized_point_info[point_id][pre_cal_camlist[1]].unsqueeze(0),
                                       parameters_dict[pre_cal_camlist[1]][7])
                calibrated_points[point_id] = \
                get_intersection(point0, point1,parameters_dict[pre_cal_camlist[0]][6:7].unsqueeze(0),
        torch.inverse(thetas_to_R(parameters_dict[pre_cal_camlist[0]][:3].unsqueeze(0), True)),
        parameters_dict[pre_cal_camlist[0]][3:6].unsqueeze(0),
        parameters_dict[pre_cal_camlist[1]][6:7].unsqueeze(0),
        torch.inverse(thetas_to_R(parameters_dict[pre_cal_camlist[1]][:3].unsqueeze(0), True)),
        parameters_dict[pre_cal_camlist[1]][3:6].unsqueeze(0))[0][:, :2]

    return calibrated_points

class ground_1camloss():
    """camera matrix의 constraint, correspondence constraint를 표현하는 loss function을 구한다."""
    def __init__(self, map_points, normalized_cam_point):
        self.map_points = map_points
        self.normalized_cam_point = normalized_cam_point

    def loss(self, parameters):
        camera_matrix,K = torch.split(parameters,[12,1])
        camera_matrix = camera_matrix.view(3,4)
        homography_matrix = camera_matrix[:,[0,1,3]]
        est_cam_points = torch.matmul(homography_matrix.unsqueeze(0),
                                    F.pad(self.map_points, [0, 1], value=1).unsqueeze(2)).squeeze(2)
        est_cam_points = est_cam_points[:, :2] / est_cam_points[:, 2:]
        undistorted_cam_point = distort_point(self.normalized_cam_point, K)

        normalizing_factor = normalize_points(undistorted_cam_point)[1]

        loss_point = torch.sum((undistorted_cam_point-est_cam_points)**2)/normalizing_factor**2

        KR = camera_matrix[:, :3]
        u, s, v = torch.svd(KR)
        s_prime = torch.cat([(s[:1] + s[1:2]) / 2, (s[:1] + s[1:2]) / 2, s[2:]], dim=0)
        KR_prime = torch.matmul(u, torch.matmul(torch.diag_embed(s_prime),v.permute(1, 0)))
        KR = KR/torch.sqrt(torch.sum(KR**2))
        KR_prime = KR_prime / torch.sqrt(torch.sum(KR_prime ** 2))
        loss_P = torch.sum((KR-KR_prime)**2)

        loss = loss_P + loss_point
        return loss

def camera_matrix_to_caminfo(camera_matrix, norm_factor_map, center_for_norm_map):
    """camera_matrix에서 theta,t,f를 구한다."""
    camera_matrix = camera_matrix.view(3, 4)
    KR = camera_matrix[:, :3]
    u, s, v = torch.svd(KR)
    s = torch.cat([(s[:1] + s[1:2]) / 2, (s[:1] + s[1:2]) / 2, s[2:]], dim=0)
    R = torch.matmul(u, v.permute(1, 0))

    R_left = thetas_to_R(R_to_thetas(R, True).unsqueeze(0), True).squeeze(0)
    if torch.sum((R - R_left) ** 2) > 1e-3:
        R = -R
        camera_matrix = -camera_matrix

    Rt = -camera_matrix[:, 3:] / s.unsqueeze(1)
    t = torch.matmul(torch.inverse(R), Rt).squeeze(1)
    if t[2]<0:
        R = torch.cat([-R[:,:2],R[:,2:]],dim=1)
        t = torch.cat([t[:2], -t[2:]], dim=0)
    t = t * norm_factor_map + F.pad(center_for_norm_map.squeeze(0), [0, 1])
    f = s[1:2] / s[2:]
    thetas = R_to_thetas(R, True)
    return thetas,t,f

def world_to_cam(point, t, R, f):
    point = torch.matmul(R, (point - t).unsqueeze(2)).squeeze(2)
    point = point[:, :2] / point[:, 2:]
    point = f * point
    return point

class make_loss_for_n():
    """n_camera에 대한 constraint들을 표현한 loss를 구하는 함수"""
    def __init__(self, normalized_saved, cam_list, n_cam, normalized_map_point_info, normalized_point_info, floor_list):
        self.normalized_saved = normalized_saved
        self.cam_list = cam_list
        self.n_cam = n_cam
        self.normalized_map_point_info = normalized_map_point_info

        self.points_for_norm_factor = []
        self.n_points_incam = []
        self.camid_to_seq = {}
        for index, camid in enumerate(self.cam_list):
            self.camid_to_seq[camid] = index
            self.n_points_incam.append(len(self.normalized_saved[camid].keys()))
            for pointid in sorted(self.normalized_saved[camid].keys()):
                self.points_for_norm_factor.append(self.normalized_saved[camid][pointid])
        self.points_for_norm_factor = torch.stack(self.points_for_norm_factor, dim=0)

        self.points_for_maploss = []
        self.map_points = []
        for point_id in sorted(normalized_map_point_info.keys()):
            for camid in sorted(set(normalized_map_point_info[point_id].keys()).intersection(self.cam_list)):
                self.points_for_maploss.append(normalized_map_point_info[point_id][camid])
                self.map_points.append(torch.tensor(normalized_map_point_info[point_id][self.n_cam], dtype=torch.float32))
        self.points_for_maploss = torch.stack(self.points_for_maploss, dim=0)
        self.map_points = F.pad(torch.stack(self.map_points, dim=0), [0, 1])

        self.points_for_mainloss0 = []
        self.points_for_mainloss1 = []
        self.is_not_floor = []
        self.cam_index_for_mainloss0 = []
        self.cam_index_for_mainloss1 = []
        for point_id in sorted(normalized_point_info.keys()):
            temp_camlist = sorted(set(normalized_point_info[point_id].keys()).intersection(self.cam_list))
            for camid0 in range(len(temp_camlist) - 1):
                for camid1 in range(camid0 + 1, len(temp_camlist)):
                    real_camid0 = temp_camlist[camid0]
                    real_camid1 = temp_camlist[camid1]
                    self.points_for_mainloss0.append(normalized_point_info[point_id][real_camid0])
                    self.points_for_mainloss1.append(normalized_point_info[point_id][real_camid1])
                    self.cam_index_for_mainloss0.append(self.camid_to_seq[real_camid0])
                    self.cam_index_for_mainloss1.append(self.camid_to_seq[real_camid1])
                    if point_id in floor_list:
                        self.is_not_floor.append(0)
                    else:
                        self.is_not_floor.append(1)
        self.is_not_floor = torch.tensor(self.is_not_floor, dtype=torch.float32).unsqueeze(1)
        self.points_for_mainloss0 = torch.stack(self.points_for_mainloss0, dim=0)
        self.points_for_mainloss1 = torch.stack(self.points_for_mainloss1, dim=0)
        self.cam_index_for_mainloss0 = torch.tensor(self.cam_index_for_mainloss0, dtype=torch.int64)\
            .unsqueeze(0).unsqueeze(2).expand(1,len(self.cam_index_for_mainloss0),24)
        self.cam_index_for_mainloss1 = torch.tensor(self.cam_index_for_mainloss1, dtype=torch.int64)\
            .unsqueeze(0).unsqueeze(2).expand(1,len(self.cam_index_for_mainloss1),24)

    def loss(self, parameters):
        parameters = parameters.view(-1, 8)
        thetas, t, f, K = torch.split(parameters, [3, 3, 1, 1], dim=1)
        R = thetas_to_R(thetas, True)
        R_inv = torch.inverse(R)
        normalizing_factor = []
        current_index = 0
        for index in range(len(self.cam_list)):
            next_index = current_index+self.n_points_incam[index]
            undistorted_points = distort_point(self.points_for_norm_factor[current_index:next_index], K[index])
            current_index = next_index
            normalizing_factor.append(normalize_points(undistorted_points)[1])
        normalizing_factor = torch.stack(normalizing_factor,dim=0)

        infos = torch.cat([R.view(-1,9),R_inv.contiguous().view(-1,9),t,f,K,normalizing_factor],dim=1)

        infos_for_map_loss = []
        for point_id in sorted(self.normalized_map_point_info.keys()):
            for camid in sorted(set(self.normalized_map_point_info[point_id].keys()).intersection(set(self.cam_list))):
                infos_for_map_loss.append(infos[self.camid_to_seq[camid]])
        infos_for_map_loss = torch.stack(infos_for_map_loss,dim=0)
        R_for_map_loss, R_inv_for_map_loss, t_for_map_loss, f_for_map_loss, K_for_map_loss, \
        norm_factor_for_map_loss = torch.split(infos_for_map_loss, [9, 9, 3,1,1,1], dim=1)
        R_for_map_loss = R_for_map_loss.view(-1,3,3)

        points_for_maploss = distort_point(self.points_for_maploss, K_for_map_loss)
        map_point_cam = torch.matmul(R_for_map_loss, (self.map_points - t_for_map_loss).unsqueeze(2)).squeeze(2)
        map_point_cam = map_point_cam[:,:2] / map_point_cam[:, 2:]
        map_point_cam = f_for_map_loss*map_point_cam
        loss_cam = (map_point_cam - points_for_maploss)/norm_factor_for_map_loss

        infos_for_mainloss0 = torch.gather(infos.unsqueeze(1).expand(infos.shape[0],
                    self.cam_index_for_mainloss0.shape[1],infos.shape[1]), 0,self.cam_index_for_mainloss0).squeeze(0)
        infos_for_mainloss1 = torch.gather(infos.unsqueeze(1).expand(infos.shape[0],
                    self.cam_index_for_mainloss1.shape[1],infos.shape[1]), 0,self.cam_index_for_mainloss1).squeeze(0)

        R_for_mainloss0, R_inv_for_mainloss0, t_for_mainloss0, f_for_mainloss0, K_for_mainloss0, \
        norm_factor_for_mainloss0 = torch.split(infos_for_mainloss0, [9, 9, 3,1,1,1], dim=1)
        R_for_mainloss1, R_inv_for_mainloss1, t_for_mainloss1, f_for_mainloss1, K_for_mainloss1, \
        norm_factor_for_mainloss1 = torch.split(infos_for_mainloss1, [9, 9, 3,1,1,1], dim=1)
        R_for_mainloss0 = R_for_mainloss0.view(-1,3,3)
        R_inv_for_mainloss0 = R_inv_for_mainloss0.view(-1, 3, 3)
        R_for_mainloss1 = R_for_mainloss1.view(-1, 3, 3)
        R_inv_for_mainloss1 = R_inv_for_mainloss1.view(-1, 3, 3)
        points_for_mainloss0 = distort_point(self.points_for_mainloss0, K_for_mainloss0)
        points_for_mainloss1 = distort_point(self.points_for_mainloss1, K_for_mainloss1)
        intersection,_,_,_ = get_intersection(points_for_mainloss0, points_for_mainloss1, f_for_mainloss0,
                    R_inv_for_mainloss0, t_for_mainloss0, f_for_mainloss1, R_inv_for_mainloss1, t_for_mainloss1)
        intersection = torch.cat([intersection[:,:2],intersection[:,2:]*self.is_not_floor],dim=1)
        intersection_cam0 = world_to_cam(intersection, t_for_mainloss0, R_for_mainloss0, f_for_mainloss0)
        intersection_cam1 = world_to_cam(intersection, t_for_mainloss1, R_for_mainloss1, f_for_mainloss1)
        loss0 = (points_for_mainloss0 - intersection_cam0)/norm_factor_for_mainloss0
        loss1 = (points_for_mainloss1 - intersection_cam1)/norm_factor_for_mainloss1
        loss = torch.sum(loss_cam**2)+torch.sum(loss0**2)+torch.sum(loss1**2)
        return loss
