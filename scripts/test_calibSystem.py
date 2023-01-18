import sys
from pathlib import Path

import cv2
import numpy as np

# Include the root directory of the project
ABS_FILE_PATH = Path(__file__).parent.absolute()
print(ABS_FILE_PATH)
sys.path.insert(0, str(ABS_FILE_PATH.parent))
print(sys.path)

from src.calibSystem import CalibSystem
from src.utils import undistort_points


def main():
    calib = CalibSystem()

    # Testing with HMS DOSAN

    # Load the images

    base_path = Path('/Users/philipp/deepingsource/aws_s3/deepingsource-calibration/3rd_floor_convenience_store')
    img_path = base_path / 'images'

    # add all images to calib
    for img_path in sorted(img_path.iterdir()):
        if str(img_path).endswith('.DS_Store'):
            continue
        calib.add_cam(path_to_img=str(img_path))

    # Load the minimap
    minimap_path = base_path / 'blueprint.png'
    calib.set_minimap(path_to_minimap=minimap_path)
    minimap = cv2.imread(str(minimap_path))

    # Load the points
    points_path = base_path / 'points' / '20221228_092419' / 'cam.npy'
    calib.load_cam_points_legacy(path_to_points=points_path)

    cam_id_0 = '192.168.1.18.png'
    cam_id_1 = '192.168.1.37.png'
    # Calibrate intrinsics of 2 cameras
    calib.optimize_intrinsics_2_cameras(cam_id_0=cam_id_0, cam_id_1=cam_id_1)

    # Get homography for one cam
    cam0_cam2minimap = calib.cam2minimap_from_points(cam_id=cam_id_0)
    # if cam1_cam2minimap is None:
    #     return

    # Try to get the homography from the neigboring cam
    cam1_cam2minimap = calib.cam2minimap_from_neighboring_camera(cam_id=cam_id_1, cam_id_neighbor=cam_id_0)

    img_0_path = base_path / 'images' / cam_id_0
    img_0 = cv2.imread(str(img_0_path))

    img_1_path = base_path / 'images' / cam_id_1
    img_1 = cv2.imread(str(img_1_path))

    img_0, minimap0 = plot_points(img_0, minimap, cam_id_0, calib)

    # Display
    cv2.imshow('Img0', img_0)
    cv2.imshow('Minimap0', minimap0)

    img_1, minimap1 = plot_points(img_1, minimap, cam_id_1, calib)

    # Display
    cv2.imshow('Img1', img_1)
    cv2.imshow('Minimap1', minimap1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_points(img, minimap, cam_id, calib):

    h, w, d = img.shape

    # Point grid
    x = np.linspace(0, w, 30)
    y = np.linspace(0, h, 30)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    pts = np.array(grid).reshape(-1, 1, 2)

    undist_pts = undistort_points(pts, calib.cameras[cam_id].K, calib.cameras[cam_id].D, mode='fisheye')

    dst = cv2.perspectiveTransform(undist_pts, calib.cameras[cam_id].cam2minimap)

    for p_pts, p_dst in zip(undist_pts, dst):
        x, y = p_pts[0]
        img = cv2.circle(img, np.int32((x, y)), radius=4, color=(0, 255, 0), thickness=-1)

        x, y = p_dst[0]
        minimap = cv2.circle(minimap, np.int32((x, y)), radius=4, color=(0, 255, 0), thickness=-1)

    return img, minimap


if __name__ == '__main__':
    main()
