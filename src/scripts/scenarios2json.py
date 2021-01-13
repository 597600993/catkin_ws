import json
import numpy as np
from random import randint
import rospy
import math
from ret2json import *
import Configs
import pika
import time


class INFO2JSON(object):
    def __init__(self):
        self.class_name = ['other', 'car', 'truck', 'bus', 'trailer',
                           'construction_vehicle',
                           'pedestrian', 'motorcycle',
                           'bicycle', 'traffic_cone', 'barrier']
        self.static_num = []
        self.static_position = []
        self.mq_channel = None
        self.mq_conn_flag = False
        self.rets = None
        self.init_connect_mq()

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

    def transfer_data_mq(self, ret):
        try:
            if self.mq_conn_flag:
                self.rets = self.to_json(ret)
                self.mq_channel.basic_publish(exchange=Configs.mq_exchange_name,
                                              routing_key=Configs.mq_routing_key,
                                              body=str(self.rets),
                                              properties=pika.BasicProperties(delivery_mode=2))
                print(self.rets)
                print(" ************** MQ Is Connecting ************** ")
        except ConnectionError as e:
            print(e)
            self.init_connect_mq()

    def add_detect_obj(self, detections, results, lon, lat, alt, camera_roty, camera):
        hdqs = [results['rot_y'] + (camera_roty / 180 * np.pi),
                results['rot_y'] + (camera_roty / 180 * np.pi) + np.pi,
                (results['rot_y'] % np.pi + (camera_roty / 180 * np.pi)) % (2 * np.pi),
                # 车辆起步,前车碰撞,盲区行人,T
                ((results['rot_y'] % np.pi + (camera_roty / 180 * np.pi)) + np.pi) % (2 * np.pi)
                # 避障车辆,横穿马路
                ]

        ptctype, vehicleclass = calculate_id(results['class'])
        longitude, latitude, altitude = lon, lat, alt
        lon = round(longitude * 10000000)
        lat = round(latitude * 10000000)
        alt = round(altitude / 0.1)
        heading = round(math.degrees(hdqs[camera]) / 0.0125)
        pos = {
            "longitude": lon,
            "latitude": lat,
            "altitude": alt
        }
        obj_size = {
            "width": round(results['dim'][1] * 100),
            "length": round(results['dim'][2] * 100),
            "height": round(results['dim'][0] * 20)
        }

        ret = {
            "ptcType": ptctype,
            "ptcId": str(results['tracking_id']),
            "plateNo": "",
            "pos": pos,
            "speed": 0,
            "heading": heading,
            "size": obj_size,
            "vehicleClass": vehicleclass
        }
        detections["participants"].append(ret)
        return detections

    def to_json(self, results, camera_gps=Configs.camera_info[0], camera_roty=Configs.camera_info[1], thresh=0.3):
        if results is not None:
            msg_id = randint(10000000, 99999999)
            lidar_time = int(rospy.get_param('get_lidar_time')) // 1000000
            detections = {
                "id": msg_id,
                "refPos": camera_gps,
                "time": lidar_time,
                "participants": []}
            obj = 0
            for t in range(len(results)):
                for k in results[t]:
                    if isinstance(results[t][k], (np.ndarray, np.float32)):
                        results[t][k] = results[t][k].tolist()
                if results[t]['score'] > thresh:
                    center_x, center_z = results[t]['loc'][0] / Configs.camera_info[2][0], \
                                         results[t]['loc'][2] / Configs.camera_info[2][1]
                    lon, lat, alt = get_distance_point(camera_gps, camera_roty, center_x,
                                                       center_z)
                    if self.class_name[results[t]['class']] is 'car':
                        detections = self.add_detect_obj(
                            detections, results[t], lon, lat, alt, camera_roty, 2)
                    if self.class_name[results[t]['class']] is 'pedestrian':
                        detections = self.add_detect_obj(
                            detections, results[t], lon, lat, alt, camera_roty, 2)

                    '''Run in Car_Missing'''
                    # if self.class_name[results[t]['class']] is 'car':
                    #     if car_list <= 1 and results[t]['tracking_id'] not in self.obj_id:
                    #         self.add_car_obj(results[t]['tracking_id'], lon, lat, alt)
                    #         rospy.set_param('car_list', car_list + 1)
                    #     if results[t]['tracking_id'] in self.obj_id:
                    #         if results[t]['tracking_id'] == 1:
                    #             detections = self.add_detect_obj(
                    #                 detections, results[t], lon, lat, alt, camera_roty, 2, "123456")
                    #         else:
                    #             detections = self.add_detect_obj(
                    #                 detections, results[t], lon, lat, alt, camera_roty, 2)

                    '''Run in Workspace'''
                    # if self.class_name[results[t]['class']] is 'traffic_cone':
                    #     if len(self.static_position) <= 4:
                    #         center_x, center_z = results[t]['loc'][0] / 1., results[t]['loc'][2] / 1.
                    #         lon, lat, alt = get_distance_point(camera_gps, camera_roty, center_x, center_z)
                    #         self.static_position.append([results[t], lon, lat, alt])
                    #         continue

                    # if self.class_name[results[t]['class']] is 'traffic_cone':
                    #     center_x, center_z = results[t]['loc'][0] / 1., results[t]['loc'][2] / 1.1
                    #     lon, lat, alt = self.get_distance_point(camera_gps, camera_roty, center_x, center_z)
                    #     if results[t]['tracking_id'] not in self.static_id:
                    #         self.add_static_obj(results[t]['tracking_id'], lon, lat, alt)
                    #         detections = self.add_detect_obj(detections, results[t], lon, lat, alt, camera_roty, 2)
                    #     else:
                    #         for p in self.static_position:
                    #             if p[0] == results[t]['tracking_id']:
                    #                 detections = self.add_detect_obj(detections, results[t], p[1], p[2], p[3],
                    #                                                  camera_roty, 2)
                    #                 break
                    obj += 1
            for static in self.static_position:
                obj += 1
                detections = self.add_detect_obj(detections, static[0], static[1], static[2], static[3],
                                                 camera_roty, 3)
            json_result = json.dumps(detections, indent=4, ensure_ascii=False)
            return json_result


if __name__ == '__main__':
    pass
