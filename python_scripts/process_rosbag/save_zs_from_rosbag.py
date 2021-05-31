#!/usr/bin/env python
import rosbag
import numpy as np
import h5py
import os 
from pyquaternion import Quaternion

def obtain_keypoints_mean_shape():

    car_mean_shape = np.array([-0.568, 0.568, 0.482, -0.482, -0.582,
        0.582, 0.702, -0.702, -0.805, -0.805,
        0.805, 0.805,
        -0.253, -0.253, 1.570, 1.570, -1.988,
        -1.988, 1.961, 1.961, -1.286, 1.355,
        -1.286, 1.355,
        1.331, 1.331, 1.331, 1.331, 0.702,
        0.702, 0.924, 0.924, 0.329, 0.329,
        0.329, 0.329])

    car_mean_shape = np.reshape(car_mean_shape, (3, 12)).T

    return car_mean_shape

def obtain_ellipsoid_shape():

    u = np.array([[1.6, 3.9, 1]])

    return u 

def normalize_uv(K, u, v):

    u_norm = (u - K[0, 0]) / K[0, 2]
    v_norm = (v - K[1, 1]) / K[1, 2]

    return u_norm, v_norm

def obtain_extrinsic():

    iTo = np.array([1,0,0, 0.3, 0,1,0, 0.0, 0,0,1, 0.0, 0,0,0,1])
    iTo = np.reshape(iTo, (4, 4))

    return iTo

def Kabsch(s_cluster, t_cluster):

    # Kabsch algorithm to estimate rotation and translation between two associated pointset.
    # step 1. Calculate the mass centeroid for both clusters. Minus it to get the delta_clusters
    # (Note that here we use column vectors)

    s_mean=np.array([np.mean(s_cluster,axis=0)])
    t_mean=np.array([np.mean(t_cluster,axis=0)])
    delta_s=(s_cluster-s_mean).T
    delta_t=(t_cluster-t_mean).T

    # step 2. Formulate the M matrix, calculate the SVD, and calculate R
    M_T=np.dot(delta_s,delta_t.T)
    M=M_T.T
    Z,sigma,LT=np.linalg.svd(M)
    icp_R=np.dot(Z,np.dot(np.diag([1,1,np.linalg.det(np.dot(Z,LT))]),LT))

    # step 3. Use centeroid and icp_R to get translation icp_p
    icp_p = t_mean.T - np.dot(icp_R, s_mean.T)
    icp_p = np.squeeze(icp_p)

    # print("check r {}".format(icp_R))
    # print("check p {}".format(icp_p))

    return icp_R, icp_p

if __name__ == "__main__":

    root_dir = "/media/erl/disk1/orcvio/opcity_40cars/"
    input_bag = root_dir + "opcity_40cars_quad_slow_no_occlusion_old.bag"
    result_dir = "/home/erl/Workspace/orcvio-backend/src/open_orcvio_cpp/ov_core/src/tests/data/40_cars/"

    origin_kpts_topic = '/uav1/detection'
    object_id = 38
    max_frame_id = 10000000000

    # get the shape 
    kps_mean_shape = obtain_keypoints_mean_shape()
    ellipsoid_shape = obtain_ellipsoid_shape()

    # get intrinsic 
    origin_intrisic_topic = '/uav1/camera/camera_info'
    for topic, msg, t in rosbag.Bag(input_bag).read_messages(): 
        if topic == origin_intrisic_topic:
            K = msg.K
            K = np.reshape(K, (3, 3))
            break

    # get extrinsic 
    iTo = obtain_extrinsic()

    # get zs and 3d positions 
    keypoint_timestamps_all = []
    zs_all = []

    frame_id = 0
    for topic, msg, t in rosbag.Bag(input_bag).read_messages(): 
        frame_id += 1 
        if frame_id > max_frame_id:
            break 
        if topic == origin_kpts_topic:
            for detection in msg.detections:
                if detection.object_id == object_id:
                    timestamp = msg.header.stamp
                    keypoint_timestamps_all.append(timestamp)
                    zs_per_frame = np.zeros((12, 2)) 
                    keypoints_gt = np.zeros((12, 3)) 
                    for kpt in detection.kpts:
                        u, v = kpt.x, kpt.y
                        u_norm, v_norm = normalize_uv(K, u, v)

                        zs_per_frame[kpt.id, 0] = u_norm
                        zs_per_frame[kpt.id, 1] = v_norm
                        zs_all.append(zs_per_frame)

                        keypoints_gt[kpt.id, 0] = kpt.x3d
                        keypoints_gt[kpt.id, 1] = kpt.y3d
                        keypoints_gt[kpt.id, 2] = kpt.z3d 

    # get wTq
    icp_R, icp_p = Kabsch(kps_mean_shape, keypoints_gt)
    wTq = np.eye(4)
    wTq[:3, :3] = icp_R
    wTq[:3, 3] = icp_p

    # get wTo
    origin_pose_topic = "/unity_command/ground_truth/uav1/pose"
    wTo_all = []

    for topic, msg, t in rosbag.Bag(input_bag).read_messages(): 
        if topic == origin_pose_topic:
            timestamp = msg.header.stamp
            if timestamp in keypoint_timestamps_all:
                t = msg.pose.position 
                q = msg.pose.orientation 
                my_quaternion = Quaternion(q.w, q.x, q.y, q.z)
                R = my_quaternion.rotation_matrix 

                wTi_per_frame = np.eye(4)
                wTi_per_frame[:3, :3] = R 
                wTi_per_frame[:3, 3] = np.array([t.x, t.y, t.z]) 
                wTo_per_frame = np.matmul(wTi_per_frame, iTo) 
                wTo_all.append(wTo_per_frame)

    # save all frames 
    total_frame_num = len(keypoint_timestamps_all)
    for i in range(total_frame_num):
        filepath = result_dir + 'frame_' + str(i) + '.h5'
        with h5py.File(filepath, "w") as f:
            f.create_dataset("zs", data = zs_all[i])
            f.create_dataset("wTo", data = wTo_all[i])
            f.create_dataset("kps_gt_3d", data = keypoints_gt)
            f.create_dataset("mean_shape", data = kps_mean_shape)
            f.create_dataset("ellipsoid_shape", data = ellipsoid_shape)
            f.create_dataset("wTq", data = wTq)
