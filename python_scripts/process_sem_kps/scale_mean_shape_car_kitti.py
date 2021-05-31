# -*- coding: utf-8 -*-

import numpy as np 
from collections import OrderedDict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

meanShape = np.array([[ 0.51617437, -0.75177691, -0.3039477 ],
       [-0.60932379, -0.76544572, -0.30421637],
       [ 0.45497868,  0.68365761, -0.25549707],
       [-0.55948299,  0.67100908, -0.24895223],
       [ 0.35455377, -1.14587136, -0.04701334],
       [-0.44722675, -1.16373931, -0.04716975],
       [ 0.30701028,  1.07196434,  0.04324782],
       [-0.42254084,  1.06302163,  0.04533736],
       [ 0.53605279, -0.32077159,  0.16844728],
       [-0.62238627, -0.33490496,  0.16933305],
       [ 0.32350219, -0.18254988,  0.39466091],
       [-0.40764679, -0.18920614,  0.39457299],
       [ 0.29858496,  0.50629007,  0.40356634],
       [-0.39017671,  0.50714026,  0.40401154]])

# Average car dimensions (in meters)
AVG_CAR_HEIGHT = 1.5208
AVG_CAR_WIDTH = 1.6362
AVG_CAR_LENGTH = 3.8600

# Compute (rough) dimensions (length, width, height) of the car shape prior
w = np.abs(np.max(meanShape[:, 0]) - np.min(meanShape[:, 0]))
l = np.abs(np.max(meanShape[:, 1]) - np.min(meanShape[:, 1]))
h = np.abs(np.max(meanShape[:, 2]) - np.min(meanShape[:, 2]))  

# Compute (anisotropic) scaling factors for each axis so that the dimensions of
# the car shape prior become roughly equal to those of an average KITTI car.
# The canonical wireframe is defined such that the width of the car is aligned with
# the Z-axis, the height with the Y-axis, and the length with the X-axis.
sz = AVG_CAR_WIDTH / w
sy = AVG_CAR_HEIGHT / h
sx = AVG_CAR_LENGTH / l

# sz /= 2
# sx /= 2
# sy /= 2

# Scale the mean shape and the basis vectors
meanShape_scaled = np.copy(meanShape)
meanShape_scaled[:, 0] = sz * meanShape[:, 0]
meanShape_scaled[:, 1] = sx * meanShape[:, 1]
meanShape_scaled[:, 2] = sy * meanShape[:, 2]

def transform_sim_part_id(self, sim_part_id):

    # part_id = sim_part_id - 1

    if sim_part_id == 1:
        part_id = 8

    if sim_part_id == 2:
        part_id = 9

    if sim_part_id == 3:
        part_id = 10

    if sim_part_id == 4:
        part_id = 11

    if sim_part_id == 5:
        part_id = 0

    if sim_part_id == 6:
        part_id = 1

    if sim_part_id == 7:
        part_id = 2

    if sim_part_id == 8:
        part_id = 3

    if sim_part_id == 9:
        part_id = 4

    if sim_part_id == 10:
        part_id = 5

    if sim_part_id == 11:
        part_id = 6

    if sim_part_id == 12:
        part_id = 7

    return part_id

def init_kp_dict():

    # % 11 ->  'R_F_RoofTop'
    kp_trans_dict[0] = 11
    # % 10 ->  'L_F_RoofTop'
    kp_trans_dict[1] = 10
    # % 12 ->  'L_B_RoofTop'
    kp_trans_dict[2] = 12
    # % 13 ->  'R_B_RoofTop'
    kp_trans_dict[3] = 13
    # % 5  ->  'R_HeadLight'
    kp_trans_dict[4] = 5
    # % 4  ->  'L_HeadLight'
    kp_trans_dict[5] = 4
    # % 6  ->  'L_TailLight'
    kp_trans_dict[6] = 6
    # % 7  ->  'R_TailLight'
    kp_trans_dict[7] = 7
    # % 1  ->  'R_F_WheelCenter'
    kp_trans_dict[8] = 1
    # % 3  ->  'R_B_WheelCenter'
    kp_trans_dict[9] = 3
    # % 0  ->  'L_F_WheelCenter'
    kp_trans_dict[10] = 0
    # % 2  ->  'L_B_WheelCenter'
    kp_trans_dict[11] = 2

kp_trans_dict = OrderedDict()
init_kp_dict()

NUM_KEYPOINTS_CAT = 14
NUM_KEYPOINTS_STAR = 12

meanShape_scaled_starmap = np.zeros((NUM_KEYPOINTS_STAR, 3))

for i in range(NUM_KEYPOINTS_CAT):
  if i in kp_trans_dict.values():
#     kp_id = transform_sim_part_id(i)
    kp_id = list(kp_trans_dict.values()).index(i)
    
#     print("cat id {}, star id {}".format(i, kp_id))
    meanShape_scaled_starmap[kp_id, :] = meanShape_scaled[i, :]

print("starmap kps")
print(repr(meanShape_scaled_starmap.T))

fig = figure()
ax = Axes3D(fig)

m = meanShape_scaled_starmap.T
# m = meanShape_scaled.T

for i in range(12): #plot each point + it's index as text above
  ax.scatter(m[0, i],m[1, i],m[2, i], color='b') 
  ax.text(m[0, i],m[1, i],m[2, i],  '%s' % (str(i)), size=20, zorder=1,  
  color='k') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# ax.view_init(azim=90, elev=0)
ax.view_init(azim=0, elev=0)

pyplot.show()