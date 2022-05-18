import rosbag
import sys

if __name__ == '__main__':
    input_bags = sys.argv[1:-1]
    output_bag = sys.argv[-1]
    with rosbag.Bag(output_bag, 'w') as outbag:
        for input_bag in input_bags:
            for topic, msg, t in rosbag.Bag(input_bag).read_messages():
                # This also replaces tf timestamps under the assumption 
                # that all transforms in the message share the same timestamp
                if topic == "/tf" and msg.transforms:
                    outbag.write(topic, msg, msg.transforms[0].header.stamp)
                else:
                    outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
