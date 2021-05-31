#!/usr/bin/env python
from numpy.linalg import inv
import numpy as np
import os
import pandas as pd
import pykitti

import utils 

"""
Change dataset path downloaded from Kitti
The converted ground-truth file will be saved at the same directory 
of the original pose file
"""

# Change this to the directory where you store KITTI data
basedir = '/home/jamesdi1993/datasets/Kitti'

# Specify the dataset to load
sequence = '06'
date = '2011_09_30' # mapping from odometry to raw dataset 
drive = '0020'

odometry_path = os.path.join(basedir, 'odometry/dataset')
raw_data_path = os.path.join(basedir, 'raw_data')

# Read both odometry and raw datasets;
odometry_data = pykitti.odometry(odometry_path , sequence)
raw_data = pykitti.raw(raw_data_path, date, drive)

output_path = os.path.join(odometry_path, "poses/" + sequence + "_quat.txt")

T_cam0_imu = raw_data.calib.T_cam0_imu
T_imu_cam0 = inv(T_cam0_imu)
poses_w_imu = [T_imu_cam0.dot(T.dot(T_cam0_imu)) for T in odometry_data.poses]


# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

print("Shape of the ground-truth poses: " + str(len(poses_w_imu)))
print("\nShape of timestamps: " + str(len(odometry_data.timestamps)))
print("=======================================")

print("\nExample of ground-truth pose: " + str(poses_w_imu[0]))
print("\nExample of timestamps: " + str(odometry_data.timestamps[0:10]))
print("=======================================")

# extract timestamps
timestamps = np.array(map(lambda x: x.total_seconds(), odometry_data.timestamps)).reshape(-1, 1)

# print(timestamps.shape)

# extra Rotations and Translations
(R_w_imu, p_w_imu) = zip(*map(lambda pose: (pose[0:3, 0:3], pose[0:3, 3]), poses_w_imu))
print("\nExample of extracted rotation: " + str(R_w_imu[0]))
print("\nExample of extracted translation: " + str(p_w_imu[0]))

# note, here we use hamilton quaternion 
# with qx qy qz qw format 
q_w_imu = np.array(map(utils.matrix_to_quaternion, R_w_imu)) # qx qy qz qw
p_w_imu = np.array(p_w_imu)

# Construct dataset and output
dataset = pd.DataFrame(np.hstack((timestamps, p_w_imu, q_w_imu))) # (timestamp(s) tx ty tz qx qy qz qw)
dataset.to_csv(output_path, sep=' ', header=False, index=False)
 







