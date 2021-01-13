import Configs
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import numpy as np
import detector_cam_lidar
import time


class ListenR(object):
    def __init__(self):
        self.images = None
        rospy.set_param('get_lidar_data', False)
        if Configs.risk_test:
            import scenarios2json
            self.rs = scenarios2json.INFO2JSON()
            rospy.set_param('get_lidar_data', True)  # False
            rospy.set_param('get_lidar_time', str(rospy.get_rostime()))
        self.pub = rospy.Publisher('/camera_ret', String, queue_size=1)

    def listener_camera(self):
        rospy.Subscriber(Configs.camera_data_sub,
                         Image, self.camera_callback, queue_size=3, buff_size=2 ** 24)
        rospy.spin()

    def camera_callback(self, data):
        try:
            # cam_header = data.header
            self.images = np.fromstring(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
            if rospy.get_param('get_lidar_data'):
                self.det_camera()
        except Exception as e:
            print(e)

    def det_camera(self):
        start_time = time.time()
        ret = camera_.detector.run(self.images, camera_.input_meta)['results']
        # ret_cam = ret2json.camera_json(ret)
        if Configs.risk_test:
            self.rs.transfer_data_mq(ret)
        print("Camera Detect time " + str(int((time.time() - start_time) * 1000)) + ' ms')
        # self.pub.publish(str(ret_cam))


if __name__ == '__main__':
    rospy.init_node('cam_listener', anonymous=False)
    listener = ListenR()
    camera_ = detector_cam_lidar.Camera()
    listener.listener_camera()
