import numpy as np
import os
from scipy.ndimage import binary_erosion


def read_kitti_bin(path):
    # Each point is 4 float32 numbers
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points

def read_vel_to_cam(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    Rotation_line = lines[1][3:]
    R = np.array([[float(num) for num in Rotation_line.split()]]).reshape(3, 3)
    Translation_line = lines[2][3:]
    T = np.array([[float(num) for num in Translation_line.split()]]).reshape(3, 1)
    vel_to_cam = np.hstack((R, T))
    vel_to_cam = np.vstack((vel_to_cam, [0, 0, 0, 1]))
    
    return vel_to_cam

def read_cam_to_cam(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    P_rect_line = [line for line in lines if line.startswith('P_rect_02')][0]
    R_rect_line = [line for line in lines if line.startswith('R_rect_02')][0]

    values = P_rect_line.split(':')[1].strip().split()
    P_rect = np.array([float(num) for num in values]).reshape(3, 4)

    values = R_rect_line.split(':')[1].strip().split()
    R_rect = np.array([float(num) for num in values]).reshape(3, 3)
    R_rect_4x4 = np.eye(4)
    R_rect_4x4[:3, :3] = R_rect

    return P_rect, R_rect_4x4