import logging
import os, sys
import numpy as np
import logging

import pykitti

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../")

import third_party.parseTrackletXML as parseTrackletXML
import kitti_detection_helper 
import utils 
import path_def 
import se3 

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", 'INFO'))

class KittiSemDataLoader():

    def __init__(self, kitti_date, kitti_drive, end_index, 
            object_label_status):
        
        self.kitti_dataset_path = path_def.kitti_dataset_path
        
        self.kitti_date = kitti_date
        self.kitti_drive = kitti_drive

        self.start_index = 0
        self.end_index = end_index

        self.cache_path = path_def.kitti_cache_path
        self.kitti_dir = self.kitti_date + '_' + self.kitti_drive

        # for loading tracklets
        self.drive_path = self.kitti_date + '_drive_' + self.kitti_drive + '_sync'

        self.tracklet_xml_path = os.path.join(self.kitti_dataset_path, self.kitti_date,
                                    self.drive_path, "tracklet_labels.xml")

        self.corner_list = []
        self.cuboid_list, self.volume_list = [], []

        # key is frame number
        self.local_volume_dict = {}
        self.local_cuboid_dict = {}
        self.local_yaw_dict = {}
        self.local_hwl_dict = {}

        self.poses_gt = []
        self.gt_position = []
        self.gt_orientation = []

        # load KITTI dataset and extrinsics 
        self.get_dataset()
        self.K = self.dataset.calib.K_cam0
        self.load_extrinsics()
        self.get_GroundTruth()

        if object_label_status == 'tracklet_label':
            self.load_tracklet()
        elif object_label_status == 'detection_label':
            self.load_detection()
        else:
            return 

        # generate path to store groundtruth 3D bounding box 
        self.generate_gt_bbox_path()
        # generate path to store the object 3D IOU results 
        self.generate_object_eval_path()

    def get_GroundTruth(self):
        """
        load gt position and orientation
        """

        # set first pose to identity
        # first_pose = self.dataset.oxts[0].T_w_imu
        # first_pose_inv = src.se3.inversePose(first_pose)
        # do not correct the orientation
        # first_pose_inv[:3, :3] = np.eye(3)

        # do not set first pose to identity
        first_pose_inv = np.eye(4)

        for o in self.dataset.oxts:

            normalized_pose_original = first_pose_inv @ o.T_w_imu
            self.poses_gt.append(normalized_pose_original)

        # gt pose is from I to G
        for i, pose in enumerate(self.poses_gt):

            # get gt position
            gt_position = np.reshape(pose[0:3, 3], (-1, 1))

            self.gt_position.append(gt_position)

            # get gt orientation
            R_wIMU = pose[0:3, 0:3]
            self.gt_orientation.append(R_wIMU)

    def get_dataset(self):
        """
        load kitti dataset using pykitti
        """

        self.dataset = pykitti.raw(self.kitti_dataset_path, self.kitti_date, self.kitti_drive, frames = range(self.start_index, self.end_index, 1))

        LOGGER.info('Drive: ' + str(self.dataset.drive))
        LOGGER.info('Frame range: ' + str(self.dataset.frames))

    def load_extrinsics(self):

        # cam to imu T
        T_camvelo = self.dataset.calib.T_cam0_velo
        T_veloimu = self.dataset.calib.T_velo_imu

        # T_cam0_imu Take a vector from IMU frame to the cam0 frame.
        # refer to https://github.com/utiasSTARS/pykitti
        # point_velo = np.array([0,0,0,1])
        # point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

        T_cam0_imu = np.matmul(T_camvelo, T_veloimu)  
        self.oTi = T_cam0_imu
        self.iTo = se3.inversePose(self.oTi)

        # add vel to imu transformation
        self.iTv = se3.inversePose(self.dataset.calib.T_velo_imu)
        self.o2Tv = self.dataset.calib.T_cam2_velo

    def generate_gt_bbox_path(self):

        self.gt_bbox_results_path = self.cache_path + self.kitti_dir + '/gt_bboxes_results/'

        if not os.path.exists(self.gt_bbox_results_path):
            os.makedirs(self.gt_bbox_results_path)

    def generate_object_eval_path(self):

        self.pr_table_dir = self.cache_path + self.kitti_dir + '/evaluation/'

        if not os.path.exists(self.pr_table_dir):
            os.makedirs(self.pr_table_dir)

    def load_tracklet(self):
        """
        # load tracklet
        # to show 3D bounding box
        # need to use the groundtruth trajectory
        # P.S. here we average all the tracklets for one object,
        # if it is moving then it is not accurate
        """

        tracklet_all = parseTrackletXML.parseXML(self.tracklet_xml_path)

        for i, tracklet in enumerate(tracklet_all):

            h, w, l = tracklet.size

            if tracklet.objectType not in ["Car", "Van", "Truck"]:
                continue

            trackletBox = np.array([
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0, 0, 0, 0, h, h, h, h]])

            corner_sublist = []
            for translation, rotation, state, occlusion, truncation, amtOcclusion, \
                amtBorders, absoluteFrameNumber in tracklet:

                # determine if object is in the image; otherwise continue
                if truncation not in (parseTrackletXML.TRUNC_IN_IMAGE, parseTrackletXML.TRUNC_TRUNCATED):
                    continue

                # print("translation {}".format(translation))

                # re-create 3D bounding box in velodyne coordinate system
                yaw = rotation[2]  # other rotations are supposedly 0
                assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'

                # transform from camera frame to world frame
                FN = absoluteFrameNumber

                # only load bbox between start and end frame
                if FN >= self.end_index:
                    # print("FN {} end {}".format(FN, self.end_index))
                    continue

                # object to velodyne transform
                vTq = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, translation[0]],
                                [np.sin(yaw), np.cos(yaw), 0.0, translation[1]],
                                [0.0, 0.0, 1.0, translation[2]],
                                [0.0, 0.0, 0.0, 1.0]])

                wTi = np.eye(4)

                wRi = self.gt_orientation[FN]
                # note q is from G to I
                wTi[:3, :3] = wRi
                wTi[:3, 3] = np.squeeze(self.gt_position[FN])

                wTq = wTi @ self.iTv @ vTq

                # force only yaw and x,y translation
                wTq = utils.poseSE32SE2(wTq)

                cornerPosInVelo = wTq[:3, :3] @ trackletBox + np.tile(wTq[:3, 3], (8, 1)).T
                corner_sublist.append(cornerPosInVelo)

                cornerInVelo = vTq[:3, :3] @ trackletBox + np.tile(vTq[:3, 3], (8, 1)).T
                cornerInCam2 = self.o2Tv @ np.vstack((cornerInVelo, np.ones((1, 8))))
                cornerInCam2 = np.eye(3) @ cornerInCam2[:3, :]

                # used for per frame IOU evaluation
                if FN not in self.local_cuboid_dict.keys():
                    self.local_cuboid_dict[FN] = [cornerInCam2.T]
                    self.local_volume_dict[FN] = [h * w * l]
                    self.local_yaw_dict[FN] = [yaw]
                    self.local_hwl_dict[FN] = [[h, w, l]]
                else:
                    self.local_cuboid_dict[FN].append(cornerInCam2.T)
                    self.local_volume_dict[FN].append(h * w * l)
                    self.local_yaw_dict[FN].append(yaw)
                    self.local_hwl_dict[FN].append([h, w, l])

            if len(corner_sublist) > 0:

                # for plotting
                corner_sublist = np.concatenate([corner_sublist], axis=0)
                corner_sub = np.mean(corner_sublist, axis=0)
                self.corner_list.append(corner_sub)

                # for 3D IOU eval
                # for global cuboids
                self.cuboid_list.append(np.mean(np.array(corner_sublist), axis=0).T)
                self.volume_list.append(h * w * l)

        self.cuboid_list = np.array(self.cuboid_list)
        self.volume_list = np.array(self.volume_list)


    def load_detection(self):
        """
        load object bounding box labels in detection benchmark 
        """

        root_dir = self.kitti_dataset_path + 'object/'
        kitti_det_loader = kitti_detection_helper.KittiDataset(root_dir,
                            self.kitti_date, self.kitti_drive)

        type_list = ['Car', 'Van', 'Truck']

        # some of the bbox are the same one
        # need to compute average bbox
        for id, object_3d_list in enumerate(kitti_det_loader.all_object_3d):
            for object_3d in object_3d_list:

                corner_sublist = []

                if object_3d.cls_type not in type_list:
                    continue

                trackletBox, oTq, yaw = object_3d.generate_corners3d()
                FN = kitti_det_loader.img_idx_list[id]

                # only load bbox between start and end frame
                if FN >= self.end_index:
                    # print("FN {} end {}".format(FN, self.end_index))
                    continue

                wTi = np.eye(4)

                wRi = self.gt_orientation[FN]
                # note q is from G to I
                wTi[:3, :3] = wRi
                wTi[:3, 3] = np.squeeze(self.gt_position[FN])

                wTq = wTi @ self.iTo @ oTq

                # force only yaw and x,y translation
                wTq = utils.poseSE32SE2(wTq)

                cornerPosInVelo = wTq[:3, :3] @ trackletBox + np.tile(wTq[:3, 3], (8, 1)).T
                corner_sublist.append(cornerPosInVelo)

                cornerPosInCam2 = oTq[:3, :3] @ trackletBox + np.tile(oTq[:3, 3], (8, 1)).T
                cornerPosInCam2 = np.eye(3) @ cornerPosInCam2[:3, :]

                # used for per frame IOU evaluation
                if FN not in self.local_cuboid_dict.keys():
                    self.local_cuboid_dict[FN] = [cornerPosInCam2.T]
                    self.local_volume_dict[FN] = [object_3d.h * object_3d.w * object_3d.l]
                    self.local_yaw_dict[FN] = [yaw]
                    self.local_hwl_dict[FN] = [[object_3d.h, object_3d.w, object_3d.l]]
                else:
                    self.local_cuboid_dict[FN].append(cornerPosInCam2.T)
                    self.local_volume_dict[FN].append(object_3d.h * object_3d.w * object_3d.l)
                    self.local_yaw_dict[FN].append(yaw)
                    self.local_hwl_dict[FN].append([object_3d.h, object_3d.w, object_3d.l])

                if len(corner_sublist) > 0:

                    # for plotting
                    corner_sublist = np.concatenate([corner_sublist], axis=0)
                    corner_sub = np.mean(corner_sublist, axis=0)
                    self.corner_list.append(corner_sub)

                    # for 3D IOU eval
                    # used for global IOU
                    self.cuboid_list.append(np.mean(np.array(corner_sublist), axis=0).T)
                    self.volume_list.append(object_3d.h * object_3d.w * object_3d.l)

        self.cuboid_list = np.array(self.cuboid_list)
        self.volume_list = np.array(self.volume_list)


    def plot_all_gt_bboxes(self, axis):
        """
        draw gt bboxes from annotations
        """

        for corner_sub in self.corner_list:
            utils.draw_box(axis, corner_sub, axes=[0, 1, 2], color='blue')


if __name__ == "__main__":

    # we can use this script to load the 
    # intrinsics and extrinsics from kitti 


    # kitti_date = '2011_09_26'
    # kitti_drive = '0022'

    kitti_date = '2011_10_03'
    kitti_drive = '0027'

    start_index = 0
    end_index = 10

    # refer to https://github.com/moshanATucsd/orcvio_cpp/blob/master/eval_results/kitti_eval/eval_info.md
    # for which method to choose for each sequence 
    # object_label_status = 'tracklet_label'
    # object_label_status = 'detection_label'
    object_label_status = ''

    DL = KittiSemDataLoader(kitti_date, kitti_drive, end_index, object_label_status)

    print("intrinsics")
    print(DL.K)

    print("extrinsics")
    print(DL.oTi)
    