import numpy as np
from transforms3d.euler import mat2euler
import os, sys 
from collections import defaultdict
import math 

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../")

import utils 
import path_def 

class ObjectEvaluator(object):
    def __init__(self, data_loader, start_index, end_index):
        """
        note that we transform gt cuboids to each frame using gt poses
        and transform detected cuboids using vio poses
        in this way, we do not consider pose drift for iou
        """

        self.K = data_loader.K

        # img id related
        self.start_index = start_index
        self.end_index = end_index

        # load gt
        self.local_cuboid_dict = data_loader.local_cuboid_dict
        self.local_volume_dict = data_loader.local_volume_dict
        self.local_yaw_dict = data_loader.local_yaw_dict
        self.local_hwl_dict = data_loader.local_hwl_dict

        # load image timestamps 

        # load bbox info.
        object_2d_bbox_filename = path_def.orcvio_cache_path + "object_2d_bbox_info.txt"
        self.obtain_object_2d_bbox_info(object_2d_bbox_filename)

        # load all camera poses 
        pose_filename = path_def.orcvio_cache_path + "stamped_traj_estimate.txt"
        self.load_all_cam_poses(pose_filename, data_loader.iTo)

        # load object map server 
        object_map_dir = path_def.orcvio_cache_path + "object_map/"
        self.load_object_map_server(object_map_dir)

        # evaluate 3d iou 
        self.eval_3d_iou()

    def eval_3d_iou(self):

        # load estimated objects
        cuboids_local_estm, vols_local_estm, object_yaw_dict, \
        object_ids_dict, object_hwl_dict = \
            generate_all_cuboids(self.start_index, self.end_index,
                    self.object_map_server, self.all_cam_poses)
        
        self.cuboids_local_estm = cuboids_local_estm
        self.vols_local_estm = vols_local_estm
        self.object_yaw_dict = object_yaw_dict
        self.object_hwl_dict = object_hwl_dict
        self.object_ids_dict = object_ids_dict

        self.save_results_IOU_values = []
        self.save_results_gt_num = 0
        self.save_results_pred_num = 0
        self.save_results_PR_table = np.zeros((3, 3))

        self.evalaute()

        # if there is no object at all 
        if self.save_results_pred_num == 0:

            self.results_str_list_init = []
            self.results_3diou_mean_init = 0

            self.results_str_list_lm = []
            self.results_3diou_mean_lm = 0

            return

        self.results_3diou_mean_lm = np.mean(self.save_results_IOU_values)
        self.results_str_list_lm = self.print_results()

    def evalaute(self):
        """
        evaluate 3d iou for each object per frame
        """

        for img_index in range(self.start_index, self.end_index):

            if img_index not in self.object_2d_bbox_info:
                # no object detected in this frame
                continue

            if img_index not in self.local_cuboid_dict:
                # no gt object in this frame
                continue

            # load gt cuboids
            cuboids_gt = np.array(self.local_cuboid_dict[img_index])
            vols_gt = np.array(self.local_volume_dict[img_index])
            yaw_gt = np.array(self.local_yaw_dict[img_index])

            gt_bbox_list = project_gt_cuboid_to_image(cuboids_gt, self.K)
            object_info_list = self.object_2d_bbox_info[img_index]

            # only keep gt that has high overlap with pred
            self.keep_gt_idx = []

            # bookkeeping of predicted objects
            cuboids_pred = []
            vols_pred = []
            yaw_pred = []

            # for all objects that have 2d bbox detections
            for object_info in object_info_list:

                # get object id and bbox
                object_id = object_info[0]
                bbox = object_info[1]

                # sometimes we do not have object_id in object_ids_dict
                # since we remove camera poses outside of window and the obs in
                # remove_outdated_obs in object_lm.py
                if img_index not in self.object_ids_dict:
                    continue
                if object_id not in self.object_ids_dict[img_index]:
                    continue

                object_id_in_list = self.object_ids_dict[img_index].index(object_id)
                cuboid_pred = self.cuboids_local_estm[img_index][object_id_in_list]
                cuboids_pred.append(cuboid_pred)

                vols_pred.append(self.vols_local_estm[img_index][object_id_in_list])
                yaw_pred.append(self.object_yaw_dict[img_index][object_id_in_list])

                iou_max = 0
                object_to_gt_id = -1
                for gt_id, gt_bbox in enumerate(gt_bbox_list):
                    iou = utils.iou_2d(gt_bbox, bbox)
                    if iou > iou_max:
                        iou_max = iou
                        object_to_gt_id = gt_id

                if iou_max > utils.iou_2d_threshold:
                    self.keep_gt_idx.append(object_to_gt_id)

            # for each image
            if len(self.keep_gt_idx) == 0:
                continue

            iou_estm, corr_estm = self.cuboidIOU(img_index, cuboids_pred, yaw_pred)
            self.save_results_IOU_values.extend(iou_estm)
            # print(img_index, iou_estm, corr_estm)

            results_list, pos_level, rot_level = self.eval_precision_recall(img_index, cuboids_pred, yaw_pred, corr_estm)
            # print(results_list)

            self.pr_results_list, self.pr_pos_level, self.pr_rot_level = results_list, pos_level, rot_level

            self.save_results_gt_num += results_list[0]
            self.save_results_pred_num += results_list[1]
            self.save_results_PR_table += results_list[2]

    def cuboidIOU(self, img_index, cuboids_pred, yaw_pred):
        '''
        @INPUT:
        cuboids1 = n x 8 x 3
        vols1 = n x 1
        cuboids2 = m x 8 x 3
        vols2 = m x 1
        @OUTPUT:
        iou12 = n x 1 = intersection over union of cuboids
        correspondences12 = n x 1 = cuboid correspondences
        '''

        # remove duplicates in the list
        self.keep_gt_idx = list(dict.fromkeys(self.keep_gt_idx))

        cuboids_gt_kept = np.array(self.local_cuboid_dict[img_index])[self.keep_gt_idx]
        yaw_gt_kept = np.array(self.local_yaw_dict[img_index])[self.keep_gt_idx]

        # we can also get these directly
        # cuboids_pred = self.cuboids_local_estm[img_index]
        # yaw_pred = self.object_yaw_dict[img_index]

        cuboids1 = np.copy(cuboids_gt_kept)
        cuboids2 = np.copy(cuboids_pred)

        """
        set z to 0 
        """
        cuboids1[:, :, 2] = 0
        cuboids2[:, :, 2] = 0

        cuboids1 = cuboids1[None, ...] if cuboids1.ndim < 3 else cuboids1
        cuboids2 = cuboids2[None, ...] if cuboids2.ndim < 3 else cuboids2

        mean1 = np.mean(cuboids1, axis=1)
        mean2 = np.mean(cuboids2, axis=1)
        # print("mean1 {}".format(mean1))
        # print("mean2 {}".format(mean2))

        correspondences12 = np.argmin(np.sum((mean1[:, None, :] - mean2[None, :, :]) ** 2, axis=2), axis=1)  # n x 1
        # print("correspondences12 {}".format(correspondences12))

        iou12 = np.empty((cuboids1.shape[0],))
        for k in range(cuboids1.shape[0]):

            shape0 = self.local_hwl_dict[img_index][k]
            shape1 = self.object_hwl_dict[img_index][correspondences12[k]]

            t0 = mean1[k, :]
            t1 = mean2[correspondences12[k], :]

            yaw0 = yaw_gt_kept.tolist()[k]
            yaw1 = yaw_pred[correspondences12[k]]

            iou12[k] = utils.iou_3d(shape0, t0, yaw0, shape1, t1, yaw1)

        return iou12, correspondences12

    def print_results(self):

        """
        print results for pr table
        """

        results_str_list = []

        denom_of_gt = 1
        denom_of_pred = 1

        num_of_gt = self.save_results_gt_num
        num_of_pred = self.save_results_pred_num
        tpc = self.save_results_PR_table

        pos_level = self.pr_pos_level
        rot_level = self.pr_rot_level

        my_str = "# of gt:  {}".format(num_of_gt)
        print(my_str)
        results_str_list.append(my_str)

        my_str = "# of pred:  {}".format(num_of_pred)
        print(my_str)
        results_str_list.append(my_str)

        format_value_str = "{} m TP: {}"

        for i in range(3):
            my_str = "{} deg".format(rot_level[i])
            print(my_str)
            results_str_list.append(my_str)

            my_str = format_value_str.format(pos_level[0], tpc[i][0])
            print(my_str)
            results_str_list.append(my_str)

            my_str = format_value_str.format(pos_level[1], tpc[i][1])
            print(my_str)
            results_str_list.append(my_str)

            my_str = format_value_str.format(pos_level[2], tpc[i][2])
            print(my_str)
            results_str_list.append(my_str)

        return results_str_list

    def eval_precision_recall(self, img_index, cuboids_pred, yaw_pred, corr_estm):
        """
        :param cuboids1:
        :param yaw_gt:
        :param cuboids2:
        :param yaw_pred:
        :param corr_estm:
        :return:
        """

        # just record the no. of pr, not percentage
        # aggregate all results for all seqs by hand
        # ie we only record TP

        # for each instance
        # cal orientation error
        # cal position error
        # categorized by ... and save to counter
        # count for gt
        # count for pred
        # count under diff rot/tran

        cuboids_gt = np.array(self.local_cuboid_dict[img_index])[self.keep_gt_idx]
        yaw_gt = np.array(self.local_yaw_dict[img_index])[self.keep_gt_idx]

        num_of_gt = np.shape(cuboids_gt)[0]
        num_of_pred = np.shape(cuboids_pred)[0]
        tpc = [[0 for x in range(3)] for y in range(3)]

        # ref Visual-Inertial-Semantic Scene Representation for 3D Object Detection
        pos_level = [0.5, 1.0, 1.5]
        rot_level = [30, 45, math.inf]

        # for each associated predicted bbx
        for i in range(num_of_gt):

            gt_lmk = cuboids_gt[i, :, :]
            t0 = np.mean(gt_lmk, axis=0)
            # print("t0 {}".format(t0))
            euler0 = yaw_gt[i]

            pred_lmk_id = int(corr_estm[i])
            pred_lmk = cuboids_pred[pred_lmk_id]
            t1 = np.mean(pred_lmk, axis=0)
            # print("t1 {}".format(t1))
            euler1 = yaw_pred[pred_lmk_id]

            """
            ignore z error by placing both cuboids 
            on the same plane 
            this is for distance in pr table 
            """
            t0[2] = 0
            t1[2] = 0

            pos_error = np.linalg.norm(np.array(t0) - np.array(t1))
            """
            do NOT ignore front/back error 
            """
            # rot_error = np.abs(euler0 - euler1) * 180 / np.pi
            # rot_error %= 180
            """
            ignore front/back error 
            in this case, the yaw error cannot 
            exceed 90
            """
            rot_error = (np.abs(euler0 - euler1) % (np.pi / 2)) * 180 / np.pi
            # print("pos_error: {}  rot_error: {}".format(pos_error, rot_error))

            pos_k_list = []
            rot_k_list = []

            if pos_error <= pos_level[0]:
                pos_k_list.append(0)
            if pos_error <= pos_level[1]:
                pos_k_list.append(1)
            if pos_error <= pos_level[2]:
                pos_k_list.append(2)

            if rot_error <= rot_level[0]:
                rot_k_list.append(0)
            if rot_error <= rot_level[1]:
                rot_k_list.append(1)
            """
            ignore rotation error
            """
            rot_k_list.append(2)

            for rot_i in rot_k_list:
                for pos_i in pos_k_list:
                    tpc[rot_i][pos_i] += 1

        return [num_of_gt, num_of_pred, tpc], pos_level, rot_level

    # load bbox info.
    def obtain_object_2d_bbox_info(self, filename):
                
        self.object_2d_bbox_info = defaultdict(list)
        self.img_timestamps_list = [] 

        img_index = self.start_index

        with open(filename, "r") as ifile:
            for line in ifile:
                entries = line.strip().split()
                timestamp = float(entries[0])

                if (len(entries) > 1):
                    object_id = int(entries[1])
                    xmin = float(entries[2])
                    ymin = float(entries[3])
                    xmax = float(entries[4])
                    ymax = float(entries[5])
                    bbox = [xmin, ymin, xmax, ymax]

                if timestamp in self.img_timestamps_list:
                    pass 
                else:
                    img_index += 1
                    self.img_timestamps_list.append(timestamp)

                if (len(entries) > 1):
                    self.object_2d_bbox_info[img_index].append([object_id, bbox])

    def load_all_cam_poses(self, filename, iTo):
        
        self.all_cam_poses = {}

        with open(filename, "r") as ifile:
            for line in ifile:
                entries = line.strip().split()
                
                timestamp = float(entries[0])
                if timestamp in self.img_timestamps_list:
                    img_index = self.img_timestamps_list.index(timestamp)
                else:
                    continue 

                px, py, pz = float(entries[1]), float(entries[2]), float(entries[3])
                qx, qy, qz, qw = float(entries[4]), float(entries[5]), float(entries[6]), float(entries[7])

                wTi = np.eye(4)
                wTi[:3, 3] = np.array([px, py, pz])
                wTi[:3, :3] = utils.to_rotation([qx, qy, qz, qw])

                self.all_cam_poses[img_index] = np.matmul(wTi, iTo)

    def load_object_map_server(self, object_map_dir):

        # each object feature in object_map_server only needs two fields 
        # 1. image ids for all observations, use self.img_timestamps_list to get this   
        # 2. optimized ellipsoid shape 

        object_dict = utils.load_est_object_state(object_map_dir)
        self.object_map_server = {}

        for object_id, object_state in object_dict.items():
            
            feature_msg = {}
            feature_msg['img_id'] = []

            # check whether the object's timestamps include current image 
            for timestamp in object_state[4]:
                img_index = self.img_timestamps_list.index(timestamp)
                feature_msg['img_id'].append(img_index)

            object_feature = ObjectFeature(object_id, feature_msg, object_state[1], object_state[2], object_state[3])
            self.object_map_server[object_id] = object_feature

def get_yaw_from_R(wRq):
    """
    get yaw from rotation matrix of object pose
    note we need to add np.pi/2 for kitti
    """
    euler_angle = mat2euler(wRq)
    yaw = -euler_angle[2]
    yaw += np.pi / 2

    return yaw

def generate_all_cuboids(start_index, end_index, 
        object_map_server, all_cam_poses):
    """
    generate cuboids for all objects
    for 3d IOU eval
    """

    # key is frame id
    cuboids_local_estm = {}
    vols_local_estm = {}

    # for yaw
    object_yaw_dict = {}
    
    # to record object ids
    object_ids_dict = {}
    object_hwl_dict = {}

    for img_index in range(start_index, end_index):

        if img_index not in all_cam_poses.keys():
            continue 

        # read camera pose and transform into camera poses
        wTo = all_cam_poses[img_index]
        wRo = wTo[:3, :3]
        wPo = wTo[:3, 3]

        for object_id, object_feature in object_map_server.items():

            # check if this object is observed in this frame
            if img_index not in object_feature.feature_msg['img_id']:
                continue

            # get shape directly from ellipsoid instead of keypoints 
            lwh = object_feature.shape[1], object_feature.shape[0], object_feature.shape[2]

            cube_q = utils.lwh2box(lwh[0], lwh[1], lwh[2])

            # use optimized wTq
            wTq = object_feature.wTq

            # force only yaw and x,y translation
            wTqk = utils.poseSE32SE2(wTq)

            # compute yaw for pr table
            yaw_pred = get_yaw_from_R(wTqk[:3, :3])

            # get predicted cuboid info.
            # transform to global frame
            cube_w = (wTqk[:3, :3] @ cube_q + wTqk[:3, 3, None]).T
            # transform to camera frame
            cuboid_pred = np.matmul(wRo.T, (cube_w - np.tile(wPo, (8, 1))).T).T

            # note, hwl are defined differently here and in dataset.py
            # h, w, l = lwh[0], lwh[2], lwh[1]
            h, w, l = lwh[2], lwh[1], lwh[0]
            if img_index in object_yaw_dict:
                object_yaw_dict[img_index].append(yaw_pred)
                cuboids_local_estm[img_index].append(cuboid_pred)
                vols_local_estm[img_index].append(np.prod(lwh))
                object_ids_dict[img_index].append(object_id)
                object_hwl_dict[img_index].append([h, w, l])
            else:
                object_yaw_dict[img_index] = [yaw_pred]
                cuboids_local_estm[img_index] = [cuboid_pred]
                vols_local_estm[img_index] = [np.prod(lwh)]
                object_ids_dict[img_index] = [object_id]
                object_hwl_dict[img_index] = [[h, w, l]]

    return cuboids_local_estm, vols_local_estm, object_yaw_dict, object_ids_dict, object_hwl_dict

class ObjectFeature():

    def __init__(self, new_id, feature_msg, ellipsoid_shape, wPq, wRq):

        # An unique identifier for the feature
        self.id = new_id

        # for obs format refer to message.py
        self.feature_msg = feature_msg

        self.shape = ellipsoid_shape

        # object pose 
        self.wTq = np.eye(4)
        self.wTq[:3, :3] = wRq
        self.wTq[:3, 3] = wPq   
        
class ResultsLogger():

    def __init__(self, pr_table_dir, kitti_eva):

        self.pr_filename = pr_table_dir + 'object_eval.txt'

        self.pr_str_list_lm = kitti_eva.results_str_list_lm
        self.iou_3d_lm = kitti_eva.results_3diou_mean_lm

        self.save_3diou_pr()

    def save_3diou_pr(self):
        """
        save 3d iou and pr table in one place
        """

        with open(self.pr_filename, "w") as f:

            f.write("%s\n" % ('3d iou after lm '))
            f.write("%s\n" % (self.iou_3d_lm))

            for my_str in self.pr_str_list_lm:
                f.write("%s\n" % (my_str))

def project_gt_cuboid_to_image(cuboids_gt, K):
    """
    reproject gt cuboid to image
    note that cuboids_gt is in local frame, so
    we do not need camera poses
    to calculate 2D IOU with object bbox
    """

    gt_bbox_list = []

    for j in range(cuboids_gt.shape[0]):
        # project_points = np.matmul(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), cuboids_gt[j].T)

        project_points = np.matmul(np.eye(3), cuboids_gt[j].T)
        project_points = project_points / np.expand_dims(project_points[2, :], 0)
        project_points = np.matmul(K, project_points)
        project_points = project_points[:2, :]

        x_min = np.min(project_points[0, :])
        x_max = np.max(project_points[0, :])
        y_min = np.min(project_points[1, :])
        y_max = np.max(project_points[1, :])

        gt_bbox_list.append([x_min, y_min, x_max, y_max])

    return gt_bbox_list
