import rosbag
bag = rosbag.Bag("/media/erl/disk1/orcvio/opcity_smooth/docker_compose_opcity_20cars_quad.bag")
image_t, det_t, odom_t = [], [], []

for topic, msg, t in bag.read_messages(topics=["/falcon/cam_left/image_raw", "/starmap/keypoints", "/unity_command/ground_truth/falcon/pose"]):
    if topic == "/falcon/cam_left/image_raw": image_t.append(msg.header.stamp)
    if topic == "/starmap/keypoints": det_t.append(msg.header.stamp)
    if topic == "/unity_command/ground_truth/falcon/pose": odom_t.append(msg.header.stamp)

print("common image and det: ", len(set(image_t) & set(det_t)))
print("common odom and det:", len(set(odom_t) & set(det_t)))
print("common image and odom:", len(set(odom_t) & set(image_t)));
print("total det:", len(det_t))
print("total odom:", len(odom_t))