import os
import cv2
import glob
import numpy as np

# GREEN = (0, 255, 0)
BLACK = (255, 255, 255)

def main():
    # Camera calibration follwoing the tutorial from: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    checkerboard_width = 4
    checkerboard_height = 6

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerboard_width * checkerboard_height, 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_height, 0:checkerboard_width].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    path = "/Users/philipp/deepingsource/Projects/3rd_floor_checkerboard_images/onvif_001_minyong_front_v2/"
    path = "/Users/philipp/deepingsource/Projects/3rd_floor_checkerboard_images/onvif_003_minyong_v2/"
    path = "/Users/philipp/deepingsource/Projects/3rd_floor_checkerboard_images/onvif_004_pete_v2/"
    path = "/Users/philipp/deepingsource/Projects/3rd_floor_checkerboard_images/onvif_002_pete_front_v2/"

    images = glob.glob(path + "/*.jpg")

    for fname in images:
        print("Processing image: ", fname)
        img = cv2.imread(fname)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (checkerboard_height, checkerboard_width), flags=None)
        # If found, add object points, image points (after refining them)
        print(ret)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (checkerboard_height, checkerboard_width), corners2, ret)
            cv2.imshow('findChessboardCorners', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Store the results
    np.savez(os.path.join(path, 'calibration.npz'), ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Refine the camera matrix
    img = cv2.imread(images[0])
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('Undistorted image', dst)
    # undistort scaled
    dst_scaled = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst_scaled = dst_scaled[y:y+h, x:x+w]
    
    cv2.imshow('Undistorted scaled image', dst_scaled)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    
    cv2.waitKey(15000)

if __name__ == '__main__':
    main()