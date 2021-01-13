import Configs
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import ros_numpy
import detector_cam_lidar
import ret2json
import time


class ListenR(object):
    def __init__(self):
        self.points = None
        rospy.set_param('get_lidar_data', False)
        self.pub = rospy.Publisher('/rslidar_ret', String, queue_size=1)

    def listener_lidar(self):
        rospy.Subscriber(Configs.rslidar_data_sub,
                         PointCloud2, self.rslidar_callback, queue_size=1, buff_size=2 ** 24)
        rospy.spin()

    def rslidar_callback(self, msg):
        try:
            # lidar_header = msg.header
            rospy.set_param('get_lidar_time', str(rospy.get_rostime()))
            msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
            self.points = ret2json.get_xyz_points(msg_cloud, True)
            rospy.set_param('get_lidar_data', True)
            self.det_lidar()
        except Exception as e:
            print(e)

    def det_lidar(self):
        start_time = time.time()
        scores, dt_box_lidar, types = lidar_.run(self.points)
        ret_lidar = ret2json.lidar_json(scores, dt_box_lidar, types)
        print("Lidar  Detect time " + str(int((time.time() - start_time) * 1000)) + ' ms')
        # print(ret_lidar)
        self.pub.publish(str(ret_lidar))
        rospy.set_param('get_lidar_data', False)


if __name__ == '__main__':
    rospy.init_node('lidar_listener', anonymous=False)
    listener = ListenR()
    lidar_ = detector_cam_lidar.Lidar()
    listener.listener_lidar()
