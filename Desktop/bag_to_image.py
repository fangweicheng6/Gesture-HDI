import os
import argparse
import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    """Extract images from a ROS bag and save them with timestamp-based filenames.
    """
    parser = argparse.ArgumentParser(
        description="Extract images from a ROS bag and save them with timestamp-based filenames.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        timestr = "%.6f" % msg.header.stamp.to_sec()
        image_name = timestr + ".png"

        cv2.imwrite(os.path.join(args.output_dir, image_name), cv_img)
        print("Wrote image %s" % image_name)

    bag.close()
    return


if __name__ == '__main__':
    main()
