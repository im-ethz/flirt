import numpy as np
from PIL import Image


class Camera:
    def __init__(self, path_to_img, cam_id):
        """
        Initialize camera

        Args:
            path_to_img (str): path to image
            scale (float): scale factor for image
            resize (bool): resize image
        """
        self.path_to_img = path_to_img
        self.cam_id = cam_id
        self.points = {}
        self.points_normalized = {}

        # Init functions
        self.get_img_info()
        self.init_patameters()

    def get_img_info(
        self,
    ):
        """
        Load image from path_to_img
        """
        img = Image.open(self.path_to_img)
        w, h = img.size
        self.img_shape = (w, h, 3)
        self.img_center = np.array((w / 2.0, h / 2.0))
        self.img_norm_length = np.sqrt(np.mean(self.img_center**2))

    def init_patameters(
        self,
    ):
        # Calibration parameters
        # Set calibration flag
        self.calibrated = False

        # Camera parameters
        self.cx = self.img_center[0]
        self.cy = self.img_center[1]
        self.f = 1.0

        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0

        self.K = np.array(
            [[self.f * self.img_norm_length, 0, self.cx], [0, self.f * self.img_norm_length, self.cy], [0, 0, 1]]
        )

        # Distortion matrix
        self.D = np.array([self.k1, self.k2, self.k3, self.k4])

        # Homography
        self.cam2minimap = None

    def set_calibrated(self, flag=True):
        self.calibrated = flag

    def set_cx(self, cx):
        self.cx = cx
        self.K[0, 2] = self.cx

    def set_cy(self, cy):
        self.cy = cy
        self.K[1, 2] = self.cy

    def set_f(self, f):
        self.f = f
        self.K[0, 0] = self.f * self.img_norm_length
        self.K[1, 1] = self.f * self.img_norm_length

    def set_k1(self, k1):
        self.k1 = k1
        self.D[0] = self.k1

    def set_k2(self, k2):
        self.k2 = k2
        self.D[1] = self.k2

    def set_k3(self, k3):
        self.k3 = k3
        self.D[2] = self.k3

    def set_k4(self, k4):
        self.k4 = k4
        self.D[3] = self.k4

    def set_camera_parameters(self, cx, cy, f, k1, k2, k3, k4):
        self.set_cx(cx)
        self.set_cy(cy)
        self.set_f(f)
        self.set_k1(k1)
        self.set_k2(k2)
        self.set_k3(k3)
        self.set_k4(k4)

    def set_distortion_parameters(self, k1, k2, k3, k4):
        self.set_k1(k1)
        self.set_k2(k2)
        self.set_k3(k3)
        self.set_k4(k4)

    def add_point(self, x, y, point_id, is_floor_point):
        """
        Add point to camera
        """
        self.points[point_id] = (x, y, is_floor_point)

        # Normalize point
        normalized_x, normalized_y = self.normalize_point_by_image(x, y)
        self.points_normalized[point_id] = (normalized_x, normalized_y, is_floor_point)

    def normalize_point_by_image(self, x, y):
        """
        Normalizes points with respect to the center of the camera
        """
        normalized_x = (x - self.img_center[0]) / self.img_norm_length
        normalized_y = (y - self.img_center[1]) / self.img_norm_length
        return (normalized_x, normalized_y)
