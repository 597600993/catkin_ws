import Configs
import sys
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import time
import cv2


def camera_node():
    cap = cv2.VideoCapture(Configs.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rospy.init_node('camera', anonymous=True)
    pub = rospy.Publisher('/camera_image', Image, queue_size=10)
    rate = rospy.Rate(Configs.camera_rate)  # how many in one minute
    i = 0
    while not rospy.is_shutdown():
        try:
            ret, frame = cap.read()
            i += 1
            if not ret and Configs.video_start:
                cap = cv2.VideoCapture(Configs.video_path)
                print("*************** Restart ****************")
                i = 0
                time.sleep(2)
                continue
            image_temp = Image()
            header = Header(stamp=rospy.Time.now())
            header.frame_id = 'camera'
            image_temp.height = frame.shape[0]
            image_temp.width = frame.shape[1]
            image_temp.encoding = 'bgr8'
            image_temp.data = frame.tostring()
            image_temp.header = header
            image_temp.step = frame.shape[1] * 3
            pub.publish(image_temp)
            rate.sleep()
            print("Frame " + str(i) + ", Time " + str(int(i / fps)) +
                  " s, Image: " + str(frame.shape[0]) + "*" + str(frame.shape[1]))
        except Exception as ex:
            print(ex)


if __name__ == "__main__":
    camera_node()
