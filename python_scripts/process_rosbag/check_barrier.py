import rosbag
import sys
bag = rosbag.Bag(sys.argv[1])
barrier_data = []
for topic, msg, t in bag.read_messages(topics=["/falcon/cam_left/detection"]):
    for det in msg.detections:
        if det.class_name == 'barrier':
            barrier_data.append(dict(pos=[det.x_pos, det.y_pos, det.z_pos],
                                     kpts=[[k.id, k.x, k.y] for k in det.kpts]))
    if len(barrier_data) > 20:
        break
import yaml
print(yaml.dump(barrier_data))