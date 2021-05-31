import numpy as np
import math
import glob 

from descartes import PolygonPatch
import shapely.geometry
import shapely.affinity
from transforms3d.euler import mat2euler
from shapely.geometry import Polygon
import transforms3d as t3d

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

from se3 import *

# if compare_with_cube_slam_flag:
iou_2d_threshold = 0.7
# elif compare_with_ucla_flag:
# iou_2d_threshold = 0.1

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx, self.cy = cx, cy
        self.h, self.w = h, w
        self.angle = angle

    def get_contour(self):
        c = shapely.geometry.box(-self.w / 2.0, -self.h / 2.0, self.w / 2.0, self.h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

def process_state_for_iou(object_state):
    # generate cuboid from object state 
    # object state = [object_class, object_sizes, wPq, wRq, timestamps]

    shape = [object_state[1][2], object_state[1][0], object_state[1][1]]

    wPq = object_state[2]
    wRq = object_state[3]

    # shape order is h, w, l
    h, w, l = shape[0], shape[1], shape[2]

    trackletBox = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0, 0, 0, 0, h, h, h, h]])

    # note, size of wPq should be ,3 
    corners3d = np.matmul(wRq, trackletBox) + np.tile(wPq, (8, 1)).T
    
    yaw = t3d.euler.mat2euler(wRq, axes='rzyx')[0]

    return [shape, corners3d, wPq, yaw]

# def iou_3d(shape0, t0, yaw0, shape1, t1, yaw1, debug_mode_flag = True):
def iou_3d(shape0, t0, yaw0, shape1, t1, yaw1, debug_mode_flag = False):
    """
    calculate iou in 3d, this method already eliminates z
    ie we only consider error in xy as only xy used in RotatedRect

    We assume bboxes are on the ground(z_bottom=0) but may have different heights.
    So we pick the smaller height from the two bboxes as the intersect height,
    and times the intersection area from the Bird-Eye-View to
    get the intersected volume.
    """

    h0, w0, l0 = shape0
    h1, w1, l1 = shape1

    # bird-view iou
    rect0 = RotatedRect(t0[0], t0[1], l0, w0, yaw0)
    rect1 = RotatedRect(t1[0], t1[1], l1, w1, yaw1)

    inter_area = rect0.intersection(rect1).area

    if inter_area == 0:
        return 0 

    # print("yaw0 {}".format(yaw0))
    # print("yaw1 {}".format(yaw1))
    # print("inter_area {}".format(inter_area))

    # find intersect height
    dh = min(h0, h1)
    inter_vol = inter_area * dh

    # print("h0 {}".format(h0))
    # print("h1 {}".format(h1))

    # volume iou
    vol0 = h0 * w0 * l0
    vol1 = h1 * w1 * l1

    iou_3d = inter_vol / (vol0 + vol1 - inter_vol)

    if debug_mode_flag:

        fig = plt.figure(1, figsize=(10, 4))
        ax = fig.add_subplot(111)

        x_pos = min(rect0.cx, rect1.cx)
        y_pos = min(rect0.cy, rect1.cy)
        ax.axis('scaled')
        ax.set_xlim(x_pos - 5, x_pos + 5)
        ax.set_ylim(y_pos - 5, y_pos + 5)
        ax.text(x_pos, y_pos, str(iou_3d), fontsize=12)

        ax.add_patch(PolygonPatch(rect0.get_contour(), fc='#990000', alpha=0.7))
        ax.add_patch(PolygonPatch(rect1.get_contour(), fc='#000099', alpha=0.7))
        ax.add_patch(PolygonPatch(rect0.intersection(rect1), fc='#009900', alpha=1))

        plt.show()
        # plt.close('all')

    return iou_3d

def find_closest_object_id(t_est, all_t_gt):
    # find the object id of the closest groundtruth object     
    return min(range(len(all_t_gt)), key = lambda i: np.linalg.norm(all_t_gt[i]-t_est))

# load gt object states 
def load_gt_object_state(gt_object_map_filename):
    
    object_dict = {}

    with open(gt_object_map_filename, "r") as ifile:
        
        count = 0

        for line in ifile:
            
            if "id" in line:
                object_id = next(ifile, '').strip()
                count += 1
        
            if "class" in line:
                object_class = next(ifile, '').strip()
                count += 1

            if "sizes" in line:
                object_sizes = next(ifile, '').strip().split()
                object_sizes = [float(i) for i in object_sizes]
                count += 1
                
            if "wPq" in line:
                px = next(ifile, '').strip()
                py = next(ifile, '').strip()
                pz = next(ifile, '').strip()
                wPq = [float(px), float(py), float(pz)]
                wPq = np.array(wPq)
                count += 1
            
            if "wRq" in line:
                r1 = next(ifile, '').strip().split()
                r1 = [float(i) for i in r1]
                r1 = np.reshape(np.array(r1), (1, -1))

                r2 = next(ifile, '').strip().split()
                r2 = [float(i) for i in r2]
                r2 = np.reshape(np.array(r2), (1, -1))
                
                r3 = next(ifile, '').strip().split()
                r3 = [float(i) for i in r3]
                r3 = np.reshape(np.array(r3), (1, -1))

                wRq = np.concatenate((r1, r2, r3), axis=0)
                count += 1
            
            if (count % 5 == 0):
                object_dict[object_id] = [object_class, object_sizes, wPq, wRq]

    return object_dict

# load estimated object states 
def load_est_object_state(est_object_map_dir):

    object_dict = {}

    for filename in glob.glob(est_object_map_dir + "*.txt"):
        filename_list = filename.split('_')
        if 'LM' in filename_list:
        # if 'map/initial' in filename_list:

            with open(filename, "r") as ifile:

                count = 0

                for line in ifile:

                    if "id" in line and "object" in line:
                        object_id = next(ifile, '').strip()
                        # convert string to int 
                        object_id = int(object_id)
                        count += 1     

                    if "class" in line:
                        object_class = next(ifile, '').strip()
                        count += 1
                    
                    if "wTq" in line:
                        wTq = np.eye(4)

                        r1 = next(ifile, '').strip().split()
                        r1 = [float(i) for i in r1]
                        r1 = np.reshape(np.array(r1), (1, -1))

                        r2 = next(ifile, '').strip().split()
                        r2 = [float(i) for i in r2]
                        r2 = np.reshape(np.array(r2), (1, -1))
                        
                        r3 = next(ifile, '').strip().split()
                        r3 = [float(i) for i in r3]
                        r3 = np.reshape(np.array(r3), (1, -1))

                        temp = np.concatenate((r1, r2, r3), axis=0)
                        wTq[:3, :] = temp

                        count += 1

                    if "ellipsoid" in line:
                        ux = next(ifile, '').strip()
                        uy = next(ifile, '').strip()
                        uz = next(ifile, '').strip()

                        object_sizes = [float(ux), float(uy), float(uz)]
                        object_sizes = np.array(object_sizes)
                        count += 1

                    if "timestamps" in line: 
                        entries = next(ifile, '').strip().split()
                        timestamps = [float(timestamp) for timestamp in entries]
                        count += 1

                if (count % 5 == 0):
                    wPq = wTq[:3, 3]
                    wRq = wTq[:3, :3]
                    object_dict[object_id] = [object_class, object_sizes, wPq, wRq, timestamps]

    return object_dict


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=2)

def poseSE32SE2(T, force_z_to_zero_flag = False):

    yaw = t3d.euler.mat2euler(T[:3, :3], axes='rzyx')[0]

    if not force_z_to_zero_flag:

        # note that we keep T[2, 3] instead of force z = 0
        T = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, T[0, 3]],
                      [np.sin(yaw), np.cos(yaw), 0.0, T[1, 3]],
                      [0.0, 0.0, 1.0, T[2, 3]],
                      [0.0, 0.0, 0.0, 1.0]])

    else:

        T = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, T[0, 3]],
                      [np.sin(yaw), np.cos(yaw), 0.0, T[1, 3]],
                      [0.0, 0.0, 1.0, 0],
                      [0.0, 0.0, 0.0, 1.0]])

    return T

def iou_2d(bbox1, bbox2):
    xy_poly_1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3]), (bbox1[0], bbox1[3])])
    xy_poly_2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3]), (bbox2[0], bbox2[3])])
    xy_intersection = xy_poly_1.intersection(xy_poly_2).area

    # note, we should - xy_intersection, instead of just xy_poly_1.area + xy_poly_2.area
    iou = xy_intersection / (xy_poly_1.area + xy_poly_2.area - xy_intersection)

    return iou

def lwh2box(l, w, h):
    """
    note, this is different from lwh to cuboid function in dataset.py
    for tracklets
    """

    box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])

    return box

def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    q = np.squeeze(q)

    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
    return R

# compute the PR table used in UCLA paper 
# this is the general function 
def evaluate_precision_recall(all_translation_gt, all_yaw_gt, all_translation_pred, all_yaw_pred):

    # num_of_gt = len(all_yaw_gt)
    num_of_gt = len(all_yaw_pred)
    
    num_of_pred = len(all_yaw_pred)
    tpc = [[0 for x in range(3)] for y in range(3)]

    # ref Visual-Inertial-Semantic Scene Representation for 3D Object Detection
    pos_level = [0.5, 1.0, 1.5]
    rot_level = [30, 45, math.inf]

    # for each associated predicted bbx
    for i in range(num_of_gt):

        t0 = all_translation_gt[i]
        # print("t0 {}".format(t0))
        # euler0 = all_yaw_gt[i]
        euler0 = abs(all_yaw_gt[i])

        t1 = all_translation_pred[i]
        # print("t1 {}".format(t1))
        # euler1 = all_yaw_pred[i]
        euler1 = abs(all_yaw_pred[i])

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
        # for debugging 
        # print("object id {}".format(i))
        # print("t0 {} t1 {}".format(t0, t1))
        # print("euler0 {} euler1 {}".format(euler0, euler1))
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

"""
a more general function to print precision recall table 
"""
def print_precision_recall_results(pos_level, rot_level, results_list):

    results_str_list = []

    num_of_gt = results_list[0]
    num_of_pred = results_list[1]

    tpc = results_list[2]

    my_str = "# of gt:  {}".format(num_of_gt)
    print(my_str)
    results_str_list.append(my_str)

    my_str = "# of pred:  {}".format(num_of_pred)
    print(my_str)
    results_str_list.append(my_str)

    # precision 
    # denominator = 22
    # denominator = num_of_pred
    # recall 
    denominator = 20
    # denominator = num_of_gt

    format_value_str = "{} m TP: {}"

    for i in range(3):
        my_str = "{} deg".format(rot_level[i])
        print(my_str)
        results_str_list.append(my_str)

        # my_str = format_value_str.format(pos_level[0], tpc[i][0])
        my_str = format_value_str.format(pos_level[0], tpc[i][0]/denominator)

        print(my_str)
        results_str_list.append(my_str)

        # my_str = format_value_str.format(pos_level[1], tpc[i][1])
        my_str = format_value_str.format(pos_level[1], tpc[i][1]/denominator)

        print(my_str)
        results_str_list.append(my_str)

        # my_str = format_value_str.format(pos_level[2], tpc[i][2])
        my_str = format_value_str.format(pos_level[2], tpc[i][2]/denominator)

        print(my_str)
        results_str_list.append(my_str)

    return results_str_list


