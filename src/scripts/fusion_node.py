import ast
import Configs
import rospy
from std_msgs.msg import String
from random import randint
import pika
from ret2json import *
import time

class Server(object):
    def __init__(self):
        self.rets = None
        self.res_cam = []
        self.res_lidar = []
        self.mq_channel = None
        self.mq_conn_flag = False
        self.init_connect_mq()

    def init_res(self):
        self.res_cam = []
        self.res_lidar = []

    def init_connect_mq(self):
        """
        init rabbit mq, connect to mq and send message to queue directly
        """
        try:
            mq_username = Configs.mq_username
            mq_pwd = Configs.mq_pwd
            mq_ip_addr = Configs.mq_ip_addr
            mq_port_num = Configs.mq_port_num
            mq_vhost = Configs.mq_vhost

            mq_credentials = pika.PlainCredentials(mq_username, mq_pwd)
            mq_connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=mq_ip_addr, port=mq_port_num, virtual_host=mq_vhost,
                                          credentials=mq_credentials))
            # connect to mq channel
            self.mq_channel = mq_connection.channel()
            self.mq_channel.exchange_declare(exchange=Configs.mq_exchange_name, exchange_type='topic', durable='true')
            # self.mq_channel.queue_declare(queue='test', durable=False, arguments={'x-message-ttl': 10000})
            self.mq_conn_flag = True
            print(" ************** MQ Connect Success ************** ")
            time.sleep(3)
        except Exception as e:
            print(e)

    def transfer_data_mq(self):
        try:
            if self.mq_conn_flag:
                self.mq_channel.basic_publish(exchange=Configs.mq_exchange_name,
                                              routing_key=Configs.mq_routing_key,
                                              body=str(self.rets),
                                              properties=pika.BasicProperties(delivery_mode=2))
                print(self.rets)
                print(" ************** MQ Is Connecting ************** ")
        except ConnectionError as e:
            print(e)
            self.init_connect_mq()

    def cam_callback(self, results):
        if results.data is not None:
            results_list = ast.literal_eval(results.data)
            for txt in results_list:
                bbox = [float(txt[2]), float(txt[3]), float(txt[4]), float(txt[5])]
                self.res_cam.append({
                    "type": txt[0],
                    "id": txt[1],
                    'bbox': bbox
                })

    def lidar_callback(self, results):
        if results.data is not None:
            results_list = ast.literal_eval(results.data)
            for txt in results_list:
                loc = lidar_to_cam(txt[1], txt[2], txt[3], Configs.cam2lidar_calib)
                velo_3D = compute_velo_box_3D(txt[1], txt[2], txt[3], txt[4], txt[5], txt[6], txt[7])
                bbox = velo3D_to_imgboxes(velo_3D, Configs.camera_calib, Configs.Img_shape)
                dim = [float(txt[4]), float(txt[5]), float(txt[6])]
                self.res_lidar.append({
                    "type": txt[0],
                    "bbox": bbox,
                    "h": dim[0],
                    "w": dim[1],
                    "l": dim[2],
                    "x": loc[0],
                    "y": loc[1],
                    "z": loc[2],
                    "r": txt[7]
                })
            self.rets = fusion_output2json(self.res_lidar, self.res_cam, Configs.IOU_thresh)
            pub.publish(str(self.rets))
            self.transfer_data_mq()
            self.init_res()


def fusion_output2json(lidars, imgs, threshold):
    res_lidar2img = []
    if imgs:
        for obj_lidar in lidars:
            for obj_img in imgs:
                iou_img2lidar = calculate_iou(obj_lidar['bbox'], obj_img['bbox'])
                if iou_img2lidar > threshold:
                    res_lidar2img.append(merge(obj_lidar, obj_img))
    else:
        for obj_lidar in lidars:
            res_lidar2img.append(merge(obj_lidar))

    msg_id = randint(10000000, 99999999)
    lidar_time = int(rospy.get_param('get_lidar_time')) // 1000000
    if res_lidar2img:
        output_res = {
            "id": msg_id,
            "refPos": Configs.rs_Pos,
            "time": lidar_time,
            "participants": res_lidar2img
        }
        return output_res
    else:
        return None


if __name__ == "__main__":
    rospy.init_node('fusion_data_listener', anonymous=False)
    pub = rospy.Publisher('/fusion_results', String, queue_size=1)
    srv = Server()
    try:
        rospy.Subscriber(Configs.camera_ret_sub,
                         String, srv.cam_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(Configs.rslidar_ret_sub,
                         String, srv.lidar_callback, queue_size=1, buff_size=2 ** 24)
        print("#########  Fusion Node Start Success  #########")
        rospy.spin()
    except Exception as e:
        print(e)
