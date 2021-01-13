import sys
import os
import numpy as np

""" ################################         PROJECT PATH INIT          ###############################  """
dir_file = os.path.abspath(os.path.dirname(__file__))
dir_src = os.path.dirname(dir_file)
camera_src = dir_src + '/camera/centertrack/'
point_src = dir_src + '/rslidar/centerpoint/'
sys.path.append(dir_src)
sys.path.append(point_src)
sys.path.append(camera_src)
# print(sys.path)

""" ################################      Calibration Info & Config     ###############################  """
camera_calib = np.array([[1200, 0, 960, 0],
                         [0, 1200, 540, 0],
                         [0, 0, 1, 0]], dtype=np.float64)

cam2lidar_calib = np.array([[-0.5812945, 0.813615, 0.01128435, -0.90554398],
                            [0.16276536, 0.12985459, -0.97808242, -3.81624634],
                            [-0.79724785, -0.56671723, -0.2079121, 4.5542527]], dtype=np.float64)

Img_shape = [1080, 1920, 3]
# longitude, latitude
lidar_loc = [121.4788994, 31.2491617]
lidar_roty = 0
lidar_ID = '66666666'
RefPos = [1214788994, 32491617, 4068]
rs_Pos = {
    "longitude": RefPos[0],
    "latitude": RefPos[1],
    "altitude": RefPos[2]
}
IOU_thresh = 0.3

""" ################################       Risk Scenarios Config      ###############################  """
camera_dict = {
    "X": [[114.076793838, 30.446333466, 0], 187, [1., 1.35]],
    "Miss": [[114.076858385, 30.445763148, 0], 350, [1.2, 1.5]]
}
camera_id = "Miss"
camera_info = camera_dict[camera_id]
video_path = "/home/today/视频/风险等级评估/video/路测车辆起步-.mp4"
risk_test = True

""" ################################        Camera Info & Config        ###############################  """
# video_path = "/mnt/data/test.mp4"
camera_data_sub = "/camera_image"
camera_ret_sub = "/camera_ret"
camera_model_path = camera_src + 'models/nuScenes_3Dtracking.pth'
camera_rate = 10
video_start = True
camera_thresh = 0.3

""" ################################        Lidar Info & Config        ###############################  """

rslidar_data_sub = "/rslidar_points"
rslidar_ret_sub = "/rslidar_ret"
lidar_config_file_path = point_src + 'configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms_demo.py'
lidar_model_path = point_src + 'models/last.pth'
lidar_rate = 10
lidar_thresh = 0.3

""" ################################        Fusion Info & Config         ###############################  """
fusion_topic = "/fusion_data"

mq_username = 'fds-ft'
mq_pwd = 'Fds-ft@2020'
mq_ip_addr = '172.31.240.139'
mq_port_num = 5672
mq_vhost = 'ord-ft'
mq_routing_key = 'hdpf.arithmetic.data.origin'
mq_exchange_name = 'hdpf-local-v0-snapshot-exchange-topic'

""" ################################          Python CMD Info           ###############################  """

# os.system("python3 " + dir_src + "/scripts/camera_lis.py")
# os.system("python3 " + dir_src + "/scripts/lidar_lis.py")
