import os

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.camera import Camera
from src.point import Point
from src.utils import (
    in_front_of_both_cameras,
    my_loss,
    to_homogeneous,
    undistort_points,
)


class CalibSystem:
    def __init__(self):
        # Point structure
        # A point has two coordinates (x, y)
        # A point belongs to one or more cameras and/or a minimap
        # A point can be a floor point
        self.cameras = {}
        self.minimap = None
        self.points = {}

    def add_cam(self, path_to_img):
        """
        Add camera to the calibration system
        """
        # Infer the cam id
        cam_id = os.path.basename(path_to_img)

        if cam_id in self.cameras.keys():
            print('[-] Warning: Camera already exists')
            return
        else:
            self.cameras[cam_id] = Camera(path_to_img, cam_id)

    def remove_cam(self, cam_id):
        """
        Remove camera from the calibration system
        """
        if cam_id not in self.cameras.keys():
            print('[-] Warning: Camera id does not exist')
            return
        else:
            self.cameras.pop(cam_id)
            # Remove points from the point system
            for point_id in self.points.keys():
                if cam_id in self.points[point_id].cameras.keys():
                    self.points[point_id].cameras.pop(cam_id)

    def set_minimap(self, path_to_minimap):
        """
        Add minimap to the calibration system
        """
        cam_id = os.path.basename(path_to_minimap)

        if self.minimap is not None:
            print('[-] Warning: Minimap already exists')
        self.minimap = Camera(path_to_minimap, cam_id)

    def add_point(self, x, y, point_id, cam_id, is_floor_point):
        """
        Add point to the calibration system
        """
        # In the camera system
        if cam_id not in self.cameras.keys():
            print('[-] Warning: Camera id does not exist')
            return
        self.cameras[cam_id].add_point(x, y, point_id, is_floor_point)

        # In the point system
        if point_id not in self.points.keys():
            self.points[point_id] = Point(point_id, is_floor_point)
        self.points[point_id].set_floor_point(is_floor_point)
        self.points[point_id].set_coordinate(x, y, cam_id)
        # Get normalized coordinate
        normalized_x, normalized_y, _ = self.cameras[cam_id].points_normalized[point_id]
        self.points[point_id].set_normalized_coordinate(normalized_x, normalized_y, cam_id)

    def add_point_to_minimap(self, x, y, point_id, is_floor_point):
        """
        Add point to minimap
        """
        if self.minimap is None:
            # Minimap does not exist
            return
        self.minimap.add_point(x, y, point_id, is_floor_point)

    def remove_point_from_system(self, point_id):
        """
        Remove point from the calibration system
        """
        # Remove point
        if point_id not in self.points.keys():
            # Point id does not exist
            return
        else:
            self.points.pop(point_id)
        # Remove point from cameras
        for cam_id in self.cameras.keys():
            if point_id in self.cameras[cam_id].points.keys():
                self.cameras[cam_id].points.pop(point_id)

    def remove_point_from_camera(self, point_id, cam_id):
        """
        Remove point from camera
        """
        if cam_id not in self.cameras.keys():
            # Camera id does not exist
            return
        if point_id not in self.cameras[cam_id].points.keys():
            # Point id does not exist
            return
        self.cameras[cam_id].points.pop(point_id)

    def remove_point_from_minimap(self, point_id):
        """
        Remove point from minimap
        """
        if self.minimap is None:
            # Minimap does not exist
            return
        if point_id not in self.minimap.points.keys():
            # Point id does not exist
            return
        self.minimap.points.pop(point_id)

    def _get_matching_point_ids(self, cam_id_1, cam_id_2):
        """
        Get matching points between two cameras
        """
        # Get matching point ids
        point_ids_1 = self.cameras[cam_id_1].points.keys()
        if cam_id_2 == 'minimap':
            point_ids_2 = self.minimap.points.keys()
        else:
            point_ids_2 = self.cameras[cam_id_2].points.keys()
        matching_point_ids = list(set(point_ids_1) & set(point_ids_2))
        return matching_point_ids

    def load_cam_points_legacy(self, path_to_points):
        loaded_data = np.load(path_to_points, allow_pickle=True).item()

        # Check if floor_points are provided
        if ('point_data' in loaded_data) and ('floor_points' in loaded_data):
            # New format
            loaded_point_data = loaded_data['point_data']
            floor_points = loaded_data['floor_points']
        else:
            # Previous format
            loaded_point_data = loaded_data

        # Make sure all keys are represented as int
        point_data = {}
        for cam_key in loaded_point_data.keys():
            point_data[int(cam_key)] = {}
            for point_key in loaded_point_data[cam_key].keys():
                point_data[int(cam_key)][int(point_key)] = loaded_point_data[cam_key][point_key]

        assert max(point_data.keys()) == len(self.cameras), 'Points not matching cameras'
        # Fill the camera points
        # Assumption: The order of the cameras is the same as they were loaded previously
        cam_keys = sorted(self.cameras.keys())

        for cam_id in sorted(point_data.keys())[:-1]:  # The last key is for the minimap
            cam_key = cam_keys[cam_id]
            for point_id in point_data[cam_id].keys():
                x = point_data[cam_id][point_id][0]
                y = point_data[cam_id][point_id][1]
                is_floor_point = point_id in floor_points

                self.add_point(x, y, point_id, cam_key, is_floor_point)

        # Fill the minimap points
        cam_id = sorted(point_data.keys())[-1]
        for point_id in point_data[cam_id].keys():
            x = point_data[cam_id][point_id][0]
            y = point_data[cam_id][point_id][1]
            is_floor_point = point_id in floor_points

            self.add_point_to_minimap(x, y, point_id, is_floor_point)

    def save_cam_points_legacy(self, path_to_points):
        # TODO
        pass

    def load_cam_points(self, path_to_points):
        # TODO
        pass

    def save_cam_points(self, path_to_points):
        # TODO
        pass

    def optimize_intrinsics_2_cameras(self, cam_id_0, cam_id_1):
        # normalized_point_info, normalized_map_point_info, cam0, cam1, floor_list, n_cam
        """
        Calibrate two cameras
        Uses levenberg-marquardt algorithm
        """
        # Get the matching points of cam_id_0 and cam_id_1
        matching_ids = self._get_matching_point_ids(cam_id_0, cam_id_1)
        points_cam0 = np.array([self.cameras[cam_id_0].points[p_id] for p_id in matching_ids])
        points_cam1 = np.array([self.cameras[cam_id_1].points[p_id] for p_id in matching_ids])

        # Loss function class
        loss_fn = my_loss(
            cam0_points=points_cam0,
            cam1_points=points_cam1,
            cam0_cx=self.cameras[cam_id_0].cx,
            cam0_cy=self.cameras[cam_id_0].cy,
            cam1_cx=self.cameras[cam_id_1].cx,
            cam1_cy=self.cameras[cam_id_1].cy,
            cam0_img_norm_length=self.cameras[cam_id_0].img_norm_length,
            cam1_img_norm_length=self.cameras[cam_id_1].img_norm_length,
            cam0_mode='fisheye',
            cam1_mode='fisheye',
        )

        # TODO init parameter exploration
        # Get the initial guess
        if self.cameras[cam_id_0].calibrated:
            f_cam0 = self.cameras[cam_id_0].f
            k1_cam0 = self.cameras[cam_id_0].k1
        else:
            f_cam0 = 1.01
            k1_cam0 = 0.014

        if self.cameras[cam_id_1].calibrated:
            f_cam1 = self.cameras[cam_id_1].f
            k1_cam1 = self.cameras[cam_id_1].k1
        else:
            f_cam1 = 0.99
            k1_cam1 = 0.022

        # Set the initial parameters
        init_params = np.array([f_cam0, f_cam1, k1_cam0, k1_cam1])

        # Attempt with Levenberg Marquardt
        # Starting optimization
        print(
            '[+] Starting Optimization f_cam0: {}, f_cam1: {}, k1_cam0: {}, k1_cam1: {}'.format(
                f_cam0, f_cam1, k1_cam0, k1_cam1
            )
        )
        optim_res = least_squares(loss_fn.loss, init_params)
        optim_params = optim_res.x

        f_cam0, f_cam1, k1_cam0, k1_cam1 = optim_params
        print(
            '[+] Finished Optimization f_cam0: {}, f_cam1: {}, k1_cam0: {}, k1_cam1: {}'.format(
                f_cam0, f_cam1, k1_cam0, k1_cam1
            )
        )

        self.cameras[cam_id_0].set_f(f_cam0)
        self.cameras[cam_id_0].set_k1(k1_cam0)
        self.cameras[cam_id_0].set_calibrated(True)

        self.cameras[cam_id_1].set_f(f_cam1)
        self.cameras[cam_id_1].set_k1(k1_cam1)
        self.cameras[cam_id_1].set_calibrated(True)

    def get_rotation_translation_between_2_cameras(self, cam_id_0, cam_id_1):
        # Get the matching points of cam_id_0 and cam_id_1
        matching_ids = self._get_matching_point_ids(cam_id_0, cam_id_1)
        points_cam0 = np.array([self.cameras[cam_id_0].points[p_id] for p_id in matching_ids])
        points_cam1 = np.array([self.cameras[cam_id_1].points[p_id] for p_id in matching_ids])

        # TODO: There must be more than ? corresponding points:
        if len(points_cam0) < 4:
            return None, None

        # Cameras must be calibrated
        if not (self.cameras[cam_id_0].calibrated and self.cameras[cam_id_1].calibrated):
            print('[-] Cameras not calibrated')
            return None, None

        # Expand the points
        points_cam0 = np.expand_dims(points_cam0[:, 0:2], axis=1).astype(np.float32)
        points_cam1 = np.expand_dims(points_cam1[:, 0:2], axis=1).astype(np.float32)

        # Undistort the points
        undist_points_cam0 = undistort_points(
            points_cam0, self.cameras[cam_id_0].K, self.cameras[cam_id_0].D, mode='fisheye', normalized=True
        )
        undist_points_cam1 = undistort_points(
            points_cam1, self.cameras[cam_id_1].K, self.cameras[cam_id_1].D, mode='fisheye', normalized=True
        )

        # Compute the essential matrix
        E, _ = cv2.findEssentialMat(
            points1=undist_points_cam0,
            points2=undist_points_cam1,
            cameraMatrix1=self.cameras[cam_id_0].K,
            distCoeffs1=self.cameras[cam_id_0].D,
            cameraMatrix2=self.cameras[cam_id_1].K,
            distCoeffs2=self.cameras[cam_id_1].D,
        )

        # import pdb;pdb.set_trace()

        undist_points_cam0 = to_homogeneous(undist_points_cam0[:, 0, :])
        undist_points_cam1 = to_homogeneous(undist_points_cam1[:, 0, :])

        # Decompose the essential matrix
        # Reference: https://amroamroamro.github.io/mexopencv/matlab/cv.decomposeEssentialMat.html
        R1, R2, t = cv2.decomposeEssentialMat(E)

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in front of both cameras
        # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
        # References:
        # * https://stackoverflow.com/questions/9796077\
        # /determining-if-a-3d-point-is-in-front-of-a-pair-of-stereo-camera-given-their-es
        # * https://youtu.be/zX5NeY-GTO0?t=3378

        sol1 = in_front_of_both_cameras(undist_points_cam0, undist_points_cam1, R1, t)
        sol2 = in_front_of_both_cameras(undist_points_cam0, undist_points_cam1, R2, t)

        if sol1 and sol2:
            print(
                '[-] Something went wrong in get_rotation_translation_between_2_cameras. \
                Both solutions for R are valid. Choosing R1'
            )
            R = R1
        elif sol1:
            R = R1
        elif sol2:
            R = R2
        else:
            print(
                '[-] Something went wrong in get_rotation_translation_between_2_cameras. \
                Neither solution for R is valid.'
            )
            return None, None, False

        return R, t, True

    def cam2minimap_from_points(self, cam_id):
        """
        Compute cam2minimap homography for one camera to the minimap
        """
        # Get the matching points of cam_id and the minimap
        matching_cam_minimap_ids = self._get_matching_point_ids(cam_id, 'minimap')
        points_cam_minimap_cam = np.array([self.cameras[cam_id].points[p_id] for p_id in matching_cam_minimap_ids])
        points_cam_minimap_minimap = np.array([self.minimap.points[p_id] for p_id in matching_cam_minimap_ids])

        # Check if there are at least 4 minimap points
        if len(points_cam_minimap_minimap) < 4:
            print('[-] Not enough minimap points to compute cam2minimap homography for cam_id:', cam_id)
            return None

        if not self.cameras[cam_id].calibrated:
            print(f'[-] Camera {cam_id} not calibrated yet.')
            return None

        # Undistort the points
        points_cam_minimap_cam = np.expand_dims(points_cam_minimap_cam[:, 0:2], axis=1).astype(np.float32)
        points_cam_minimap_cam = undistort_points(
            points_cam_minimap_cam, self.cameras[cam_id].K, self.cameras[cam_id].D, mode='fisheye'
        )

        # Compute cam2minimap homography
        cam2minimap, _ = cv2.findHomography(points_cam_minimap_cam, points_cam_minimap_minimap)
        self.cameras[cam_id].cam2minimap = cam2minimap
        return cam2minimap

    def cam2minimap_from_neighboring_camera(self, cam_id, cam_id_neighbor):
        # Check if the cam2minimap homography of the neighbouring cam is available
        if self.cameras[cam_id_neighbor].cam2minimap is None:
            print('[-] Homography of neighboring cam not available')
            return None

        # Calculate rotation and translation between the two cams
        R_cam2neighbor, t_cam2neighbor, success = self.get_rotation_translation_between_2_cameras(
            cam_id_0=cam_id, cam_id_1=cam_id_neighbor
        )
        if not success:
            R_neighbor2cam, t_neighbor2cam, success = self.get_rotation_translation_between_2_cameras(
                cam_id_0=cam_id_neighbor, cam_id_1=cam_id
            )
            if success:
                R_cam2neighbor = np.linalg.inv(R_neighbor2cam)
                t_cam2neighbor = -t_neighbor2cam
            else:
                print('[-] get_rotation_translation_between_2_cameras failed')
                return None

        # Construct 2D? extrinsic matrix (for homography)
        # cam2neighbor_extrinsic = np.vstack((np.hstack((R_cam2neighbor, t_cam2neighbor)), [0,0,0,1]))
        cam2neighbor_extrinsic = np.hstack((R_cam2neighbor[:, :2], t_cam2neighbor))

        # Construct 2D projection matrix, i.e. homography matrix H (3x3)
        cam2neighbor = self.cameras[cam_id].K @ cam2neighbor_extrinsic[:3]

        cam2minimap = cam2neighbor @ self.cameras[cam_id_neighbor].cam2minimap
        self.cameras[cam_id].cam2minimap = cam2minimap
        return cam2minimap
