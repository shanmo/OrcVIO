import rosbag
import rospy
from sensor_msgs.msg import Imu

# change these three parameters
root_dir = "/home/erl/moshan/open_orcvio/catkin_ws_openvins/"
input_bag = root_dir + "data/with_keypoints_original.bag"
output_bag = root_dir + "data/with_keypoints.bag"
input_imu = root_dir + "src/open_vins/results/imu.txt"

f_imu = open(input_imu,"r")
imu_records = f_imu.readlines()
f_imu.close()
imu_pointer = 0
imu_next = imu_records[imu_pointer].split()

with rosbag.Bag(output_bag, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(input_bag).read_messages():
        if imu_pointer<len(imu_records) and t > rospy.Time.from_sec((float(imu_next[0]))):
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.from_sec((float(imu_next[0])))
            imu_msg.header.frame_id = "husky/imu_sim"
            imu_msg.angular_velocity.x = float(imu_next[1])
            imu_msg.angular_velocity.y = float(imu_next[2])
            imu_msg.angular_velocity.z = float(imu_next[3])
            imu_msg.linear_acceleration.x = float(imu_next[4])
            imu_msg.linear_acceleration.y = float(imu_next[5])
            imu_msg.linear_acceleration.z = float(imu_next[6])
            imu_pointer += 1
            if imu_pointer<len(imu_records):
                imu_next = imu_records[imu_pointer].split()
            outbag.write("husky/imu_sim", imu_msg, rospy.Time.from_sec((float(imu_next[0]))))
        if topic == "/husky/camera/detection":
            closest_t = msg.header.stamp
        if msg._has_header:
            if topic == "/husky/detection_image":
                outbag.write(topic, msg, t=closest_t)
            else:
                outbag.write(topic, msg, t=msg.header.stamp)
        else:
            outbag.write(topic, msg, closest_t)
    