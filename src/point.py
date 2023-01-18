class Point:
    def __init__(self, point_id, is_floor_point=False):
        self.point_id = point_id
        self.is_floor_point = is_floor_point
        # Dictionary of camera ids, where this point is present
        self.cameras = {}
        self.cameras_normalized = {}

    def set_coordinate(self, x, y, cam_id):
        self.cameras[cam_id] = (x, y)

    def set_normalized_coordinate(self, x, y, cam_id):
        self.cameras_normalized[cam_id] = (x, y)

    def normalize_coordinate(self, cam_id, img_center, img_norm_length):
        """
        Normalizes points with respect to the center of the camera
        """
        x, y = self.cameras[cam_id]
        normalized_x = (x - img_center[0]) / img_norm_length
        normalized_y = (y - img_center[1]) / img_norm_length
        self.cameras_normalized[cam_id] = (normalized_x, normalized_y)

    def set_floor_point(self, is_floor_point):
        self.is_floor_point = is_floor_point
