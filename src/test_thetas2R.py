import cv2
import torch
import numpy as np

from utils import thetas_to_R, eulerAnglesToRotationMatrix
from utils import R_to_thetas, rotationMatrixToEulerAngles


# Path to parameters
params_path = "/Users/philipp/deepingsource/Projects/calibrate_matrix/OFFICE_OFFICE/points/192.168.1.11_192.168.1.12_192.168.1.17_192.168.1.18_minimap_cropped/192.168.1.11.npy"

# Load the parameters
param = torch.from_numpy(np.load(params_path))

# Run Thetas to R
# Calculates the transformation from Euler angles to the rotation matrix
thetas, t, f, K = torch.split(param, [3, 3, 1, 1])
print(thetas)
R = thetas_to_R(thetas.unsqueeze(0), False).squeeze(0)
print(R)
R2 = eulerAnglesToRotationMatrix(thetas)
print(R2)

print("##### Reverse #####")
# R to theta
theta_rev = R_to_thetas(R, False)
print(theta_rev)
theta_rev2 = rotationMatrixToEulerAngles(R.numpy())
print(theta_rev2)
