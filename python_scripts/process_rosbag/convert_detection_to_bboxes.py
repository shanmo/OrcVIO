#!/usr/bin/env python
import os
import rosbag
import numpy as np 
import copy
import argparse
from geometry_msgs.msg import PoseStamped

from sort_ros.msg import TrackedBoundingBoxes

def scriptdir(f=__file__):
    return os.path.dirname(f)

def rel2abs(relpath, refdir=scriptdir()):
    return os.path.join(refdir, relpath)

if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Convert the kpts message.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--input_bag', type=str, default="/media/erl/disk1/orcvio/opcity_40cars/opcity_40cars_quad.bag", 
                        help='Input Rosbag')
    parser.add_argument('--output_bag', type=str, default='/media/erl/disk1/orcvio/opcity_40cars/opcity_40cars_quad_new.bag',
                        help='Output Rosbag')

    parser.add_argument('--origin_kpts_topic', type=str, default='/falcon/cam_left/detection',
                        help='The orignal topic name of the kpts info. /ns/detection')
    parser.add_argument('--bbox_topic', type=str, default='/sort_ros/tracked_bounding_boxes',
                        help='The topic name of the bbox info.')
    parser.add_argument('--template_bag', type=str, default='/media/erl/disk1/orcvio/rosbags_old/arl_husky_overpasscity_threecar_640480_pose_kps.bag',
                        help='A rosbag having TrackedBoundingBoxes msg inside')

    # one example is 
    # python convert_detection_to_bboxes.py --template_bag /media/erl/disk1/orcvio/mit_rosbags/mit_medfield_modified_with_keypoints.bag --input_bag /media/erl/disk1/orcvio/opcity_smooth/docker_compose_opcity_20cars_quad_no_occ_with_doors.bag --output_bag /media/erl/disk1/orcvio/opcity_smooth/docker_compose_opcity_car_door_quad_bboxes_only.bag 
    
    args = parser.parse_args()    
    input_bag = args.input_bag
    output_bag = args.output_bag
    template_bag = args.template_bag

    origin_kpts_topic = args.origin_kpts_topic
    bbox_topic = args.bbox_topic

    # copy the bboxes message 
    # the lines below are copy a template for the special topic message
    for topic, msg, t in rosbag.Bag(template_bag).read_messages(): 
        if topic == bbox_topic:
            if len(msg.bounding_boxes) == 0:
                continue
            bbox_msg_template = copy.deepcopy(msg)
            break
    bbox_msg_template.serialize(open(rel2abs('bbox_msg_template.msg'), 'w'))

    bbox_msg_template.header.frame_id = origin_kpts_topic
    det_template = bbox_msg_template.bounding_boxes[0]

    closest_t = -1

    with rosbag.Bag(output_bag, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(input_bag).read_messages():  
            # convert detection to starmap keypoint topic 
            if topic == origin_kpts_topic:
                
                bbox_msg_template.header = msg.header
                bbox_msg_template.bounding_boxes = []

                for detection in msg.detections:

                    obj_det = copy.deepcopy(det_template)
                    obj_det.xmin = detection.x_min
                    obj_det.ymin = detection.y_min
                    obj_det.xmax = detection.x_max
                    obj_det.ymax = detection.y_max
                    obj_det.id = detection.object_id
                    obj_det.Class = detection.class_name
                    obj_det.lost_flag = True

                    bbox_msg_template.bounding_boxes.append(obj_det)

                outbag.write(bbox_topic, bbox_msg_template, t=t)

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
            
            
