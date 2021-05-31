#!/usr/bin/env python
from numpy.linalg import inv
import numpy as np
import transforms3d as tf

# util function 

def matrix_to_quaternion(R):
    # convert rotation matrix to Hamilton quaternion 
    # [qx, qy, qz, qw]
    # rotation matrix to angle axis
    cos = np.clip((np.trace(R) - 1) * 0.5, -1, 1)
    theta = math.acos(cos)

    if theta < np.spacing(1):
        w = np.zeros((3, 1))
    else:
        w = 1 / (2 * math.sin(theta)) * np.array([R[2, 1] - R[1, 2],
                                                  R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        w = w.T

    theta_w = theta * w
    theta_w_norm = np.linalg.norm(theta_w)

    if theta_w_norm == 0:
        return np.array([0, 0, 0, 1])

    q_hamilton = tf.quaternions.axangle2quat(theta_w / theta_w_norm, theta_w_norm)

    q_output = np.array([q_Hamilton[1], q_Hamilton[2], q_Hamilton[3], q_Hamilton[4], q_Hamilton[0]])

    return q_output