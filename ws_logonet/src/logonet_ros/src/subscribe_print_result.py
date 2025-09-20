#!/usr/bin/env python

import rospy
from logonet_ros.msg import DetectionResult

# def detections_callback(msg):
#     rospy.loginfo("测试订阅者收到/logonet_detection话题的消息")
#     rospy.loginfo(f"Class Names: {msg.class_names}")
#     rospy.loginfo(f"Bounding Boxes: {msg.bounding_boxes}")


#     print("\n\n\n")

def detections_callback(msg):
    rospy.loginfo("测试订阅者收到/logonet_detection话题的消息")
    
    # 手势类别ID
    gesture_class_ids = msg.class_names

    # 机体坐标系下手臂框中心点
    gesture_center_point = msg.bounding_boxes

    # 打印 Class Names 和 Bounding Boxes
    print("Class Names:", gesture_class_ids)
    print("Bounding Boxes (IMU Centers):", gesture_center_point)

    print("\n\n\n")  # 空行分隔输出，便于观察



def listener():
    rospy.init_node('detection_listener', anonymous=True)
    rospy.Subscriber('/logonet_detection', DetectionResult, detections_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
