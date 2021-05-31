import rosbag
import sys

if __name__ == '__main__':
    input_bag1 = sys.argv[1]
    input_bag2 = sys.argv[2]
    output_bag = sys.argv[3]
    with rosbag.Bag(output_bag, 'w') as outbag:
        for input_bag in (input_bag1, input_bag2):
            for topic, msg, t in rosbag.Bag(input_bag).read_messages():
                # This also replaces tf timestamps under the assumption 
                # that all transforms in the message share the same timestamp
                if topic == "/tf" and msg.transforms:
                    outbag.write(topic, msg, msg.transforms[0].header.stamp)
                else:
                    outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
