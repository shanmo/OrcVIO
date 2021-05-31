#!/usr/bin/env python
import os
import rosbag
import numpy as np 
import copy
import argparse
from geometry_msgs.msg import PoseStamped

from starmap_ros_msgs.msg import TrackedBBoxListWithKeypoints

def scriptdir(f=__file__):
    return os.path.dirname(f)

def rel2abs(relpath, refdir=scriptdir()):
    return os.path.join(refdir, relpath)

if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Convert the kpts message.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--input_bag', type=str, default="/media/erl/disk1/orcvio/opcity_new/docker_compose_opcity_20cars_quad_no_occ_with_doors_barriers_smooth_full_loop_2.bag", 
                        help='Input Rosbag')
    parser.add_argument('--output_bag', type=str, default='/media/erl/disk1/orcvio/opcity_new/docker_compose_opcity_20cars_quad_no_occ_with_doors_barriers_smooth_full_loop_2_new_lost_flag.bag',
                        help='Output Rosbag')

    parser.add_argument('--origin_kpts_topic', type=str, default='/falcon/cam_left/detection',
                        help='The orignal topic name of the kpts info. /ns/detection')
    parser.add_argument('--kpts_topic', type=str, default='/starmap/keypoints',
                        help='The topic name of the kpts info.')

    # one example is 
    # python convert_detection_to_keypoints_lost_flag.py --input_bag /media/erl/disk1/orcvio/opcity_new/docker_compose_opcity_20cars_quad_no_occ_with_doors_barriers_new.bag --output_bag /media/erl/disk1/orcvio/opcity_new/docker_compose_opcity_car_door_barrier_quad.bag 
    
    args = parser.parse_args()    
    input_bag = args.input_bag
    output_bag = args.output_bag

    origin_kpts_topic = args.origin_kpts_topic
    kpts_topic = args.kpts_topic

    if os.path.exists(rel2abs('kpts_msg_template.msg')):
        kpts_msg_template = TrackedBBoxListWithKeypoints()
        kpts_msg_template.deserialize(open(rel2abs('kpts_msg_template.msg'), 'r').read())
    else:
        print("kpts_msg_template does not exist!")

    kpts_msg_template.header.frame_id = origin_kpts_topic
    bbox_template = kpts_msg_template.bounding_boxes[0].bbox
    keypoint_template = kpts_msg_template.bounding_boxes[0].keypoints[0]
    det_template = kpts_msg_template.bounding_boxes[0]

    closest_t = -1

    all_frame_bbox_ids = []
    with rosbag.Bag(output_bag, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(input_bag).read_messages():  
            # convert detection to starmap keypoint topic 
            if topic == origin_kpts_topic:
                if len(msg.detections) == 0:
                    continue 
                curr_frame_bbox_ids = set()
                for detection in msg.detections:
                    curr_frame_bbox_ids.add(detection.object_id)
                all_frame_bbox_ids.append(curr_frame_bbox_ids)

    total_frame_num = len(all_frame_bbox_ids)
    with rosbag.Bag(output_bag, 'w') as outbag:
        count = 0 
        for topic, msg, t in rosbag.Bag(input_bag).read_messages():  
            # convert detection to starmap keypoint topic 
            if topic == origin_kpts_topic:
                
                kpts_msg_template.header = msg.header
                kpts_msg_template.bounding_boxes = []

                for detection in msg.detections:

                    obj_det = copy.deepcopy(det_template)
                    obj_det.bbox.xmin = detection.x_min
                    obj_det.bbox.ymin = detection.y_min
                    obj_det.bbox.xmax = detection.x_max
                    obj_det.bbox.ymax = detection.y_max
                    obj_det.bbox.id = detection.object_id
                    obj_det.bbox.Class = detection.class_name

                    # check whether the object is lost 
                    if count < total_frame_num - 1:
                        if detection.object_id in all_frame_bbox_ids[count+1]:
                            obj_det.bbox.lost_flag = False 
                        else: 
                            obj_det.bbox.lost_flag = True

                    obj_det.keypoints = []

                    # only keep this bbox if the keypoints are not zero 
                    is_valid_bbox_flag = False 

                    for kpt in detection.kpts:

                        obj_kpt = copy.deepcopy(keypoint_template)
                        obj_kpt.x = kpt.x
                        obj_kpt.y = kpt.y

                        # we keep the bbox if at least one kp is not zero 
                        if (kpt.x != 0 and kpt.y != 0):
                            is_valid_bbox_flag = True

                        obj_kpt.semantic_part_label = kpt.id 
                        # set a dummy value since part label is not used 
                        obj_kpt.semantic_part_label_name = "dummy"
                        obj_det.keypoints.append(obj_kpt)

                    if is_valid_bbox_flag:
                        kpts_msg_template.bounding_boxes.append(obj_det)

                # outbag.write(kpts_topic, kpts_msg_template, t=msg.header.stamp)
                outbag.write(kpts_topic, kpts_msg_template, t=t)
                
                if len(msg.detections) > 0:
                    count += 1 

            else:

                if hasattr(msg, "header"):
                    # closest_t = msg.header.stamp
                    #outbag.write(topic, msg, t=msg.header.stamp)
                    closest_t = t
                    outbag.write(topic, msg, t=t)
                else:
                    if closest_t == -1:
                        continue
                    else:
                        outbag.write(topic, msg, t=closest_t)

     
            
            
