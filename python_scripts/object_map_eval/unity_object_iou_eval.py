import numpy as np 
import os, sys, glob  
from collections import OrderedDict 

import utils

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../")

import path_def 

if __name__ == "__main__":

    # debug_mode_flag = True 
    debug_mode_flag = False 

    object_map_dir = path_def.orcvio_cache_path + "object_map/"

    # prepare groundtruth object map 
    gt_object_map_filename = "gt_object_states.txt"
    object_gt_dict = utils.load_gt_object_state(object_map_dir + gt_object_map_filename)
    object_states_for_iou_gt = OrderedDict()
    for object_id, object_state in object_gt_dict.items():
        object_states_for_iou_gt[object_id] = utils.process_state_for_iou(object_state)

    # prepare estimated object map 
    object_est_dict = utils.load_est_object_state(object_map_dir)
    object_states_for_iou_est = OrderedDict()
    for object_id, object_state in object_est_dict.items():
        object_states_for_iou_est[object_id] = utils.process_state_for_iou(object_state)

    # calculate 3D IOU
    all_yaw_pred = []
    all_translation_pred = []
    all_yaw_gt = []
    all_translation_gt = []

    all_t_gt_list = []
    for object_id, object_state in object_states_for_iou_gt.items():
        all_t_gt_list.append(object_state[2])

    min_iou_thresh = 0.2 

    mean_3d_iou = 0.0 
    counted_object_num = 0

    for object_id, object_state_est in object_states_for_iou_est.items():
        
        # find the object id of the closest groundtruth object 
        t_est = object_state_est[2]
        closest_gt_object_id = utils.find_closest_object_id(t_est, all_t_gt_list)
        closest_gt_object_id_converted = list(object_states_for_iou_gt.keys())[closest_gt_object_id]
        object_state_gt = object_states_for_iou_gt[closest_gt_object_id_converted]

        shape_est, shape_gt = object_state_est[0], object_state_gt[0]
        t_gt = object_state_gt[2]
        yaw_est, yaw_gt = object_state_est[3], object_state_gt[3]

        all_yaw_pred.append(yaw_est)
        all_yaw_gt.append(yaw_gt)

        all_translation_pred.append(t_est)
        all_translation_gt.append(t_gt)

        iou3d_result = utils.iou_3d(shape_est, t_est, yaw_est, 
            shape_gt, t_gt, yaw_gt, debug_mode_flag)
        
        if iou3d_result > min_iou_thresh:
            mean_3d_iou += iou3d_result
            counted_object_num += 1 

    mean_3d_iou /= (counted_object_num + 1e-6)
    print("No. objects counted {}".format(counted_object_num))
    print("No. objects detected {}".format(len(object_est_dict)))
    print("Average 3D IOU {}".format(mean_3d_iou))

    # compute the PR table 
    results_list, pos_level, rot_level = utils.evaluate_precision_recall(all_translation_gt, all_yaw_gt, 
        all_translation_pred, all_yaw_pred)
    # print the results 
    utils.print_precision_recall_results(pos_level, rot_level, results_list)


