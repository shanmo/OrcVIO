int64 object_id

std_msgs/Float64MultiArray residual # object residual 
std_msgs/Float64MultiArray jacobian_wrt_object_state # jacobians wrt poses in the object's state
std_msgs/Float64MultiArray jacobian_wrt_sensor_state # jacobians wrt poses in the robot's state
std_msgs/Float64MultiArray valid_camera_pose_mat # valid camera poses 
float64[] timestamps # timestamps for the poses 
int32[] zs_num_wrt_timestamps # number of total semantic keypoints 