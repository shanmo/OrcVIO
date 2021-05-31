"""
This script is for loading the object 3D info. from the 
object detection dataset in KITTI 
ref: https://github.com/sshaoshuai/PointRCNN/blob/master/lib/utils/object3d.py
"""

import numpy as np
import os

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        """
        determines difficulty level
        :return:
        """
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w

        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h]])

        yaw = self.ry
        # print("check yaw {}".format(yaw))

        # convert yaw to world frame
        yaw = -yaw + np.pi/2

        cRq = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0]])

        oRc = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [0, -1, 0]])

        translation = self.pos
        # print("check trans {}".format(translation))

        oTq = np.eye(4)
        corners3d = oTq[:3, :3] @ trackletBox + np.tile(oTq[:3, 3], (8, 1)).T

        oTq[:3, :3] = oRc @ cRq
        oTq[:3, 3] = translation

        return corners3d, oTq, yaw

class KittiDataset():
    def __init__(self, root_dir, date_id, drive_id):
        self.split = 'train'
        self.label_dir = os.path.join(root_dir, 'training', 'label_2')
        self.mapping_filename = os.path.join(root_dir, 'training', 'train_mapping.txt')

        """
        read the random mapping 
        """
        self.random_index_filename = os.path.join(root_dir, 'training', 'train_rand.txt')
        self.load_rand_idx()

        self.date_id = date_id
        self.drive_id = drive_id

        self.get_img_idx()
        self.get_all_bbox()

    def load_rand_idx(self):
        """

        :return:
        """

        f_shuffle = open(self.random_index_filename, 'r')
        shuffle_index = f_shuffle.readlines()[0].split(",")
        shuffle_index = [int(index) for index in shuffle_index]
        self.mapping_index = {}
        for i in range(len(shuffle_index)):
            self.mapping_index[shuffle_index[i]-1] = i
        f_shuffle.close()

    def get_all_bbox(self):

        self.all_object_3d = []

        for idx in self.file_idx_list:
            self.all_object_3d.append(self.get_label(idx))

    def get_label(self, idx):

        label_file = os.path.join(self.label_dir, '%06d.txt' % self.mapping_index[idx])

        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_img_idx(self):

        self.file_idx_list = []
        self.img_idx_list = []
        count = 0

        full_name = self.date_id + '_' + 'drive' + '_' + self.drive_id \
                + '_' + 'sync'

        with open(self.mapping_filename, 'r') as f:

            lines = f.readlines()

            for line in lines:
                line = line.split(" ")
                if line[0] == self.date_id:
                    if line[1] == full_name:
                        self.file_idx_list.append(count)
                        self.img_idx_list.append(int(line[2]))

                count += 1






