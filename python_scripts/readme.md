## preprocessing 

### process the rosbags 

- scripts for processing the rosbags is in `process_rosbag` folder 
- `convert_detection_to_keyoints.py` is used to convert the `detection` message type in unity to `starmap/keypoints` type used in the front end 
- `add_sim_imu.py` adds the simulated IMU data from OpenVINS to the rosbag 
- `save_zs_from_rosbag` retrieves the semantic keypoint measurements and saves them for unit test 

### process the yaml file 

- `gen_gt_object_yaml.py` takes the yaml file used to spawn the object in unity and generate the object groundtruth states in yaml file located in the config folder

## trajectory evaluation 

- `trajectory_eval/batch_run_euroc.py` evaluates different parameters on EuRoC
- `trajectory_eval/traj_eval.py` evaluates the trajectory accuracy wrt groundtruth 
- `trajectory_eval/kitti_odom_gt_converter.py` converts the KITTI oxt data to groundtruth trajectory 

## object mapping evaluation 

- `object_map_eval/unity_object_iou_eval.py` evalutes the 3D IOU for object mapping, in Unity simulator 
- `object_map_eval/kitti_object_iou_eval.py` evalutes the 3D IOU for object mapping, in KITTI dataset for a single sequence 
- `object_map_eval/kitti_construct_pr_table_all_sequences.py` constructs the PR table for the paper for all the KITTI sequences  
