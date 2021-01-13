import numpy as np
import Configs
import geopy
from geopy.distance import distance
import math
from pyquaternion import Quaternion
from sensor_msgs.msg import PointCloud2, PointField

class_name = ['other',
              'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
              'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0, 0, 1], radians=yaw)


def camera_json(results):
    cam_obj = []
    if results is not None:
        for t in range(len(results)):
            for k in results[t]:
                if isinstance(results[t][k], (np.ndarray, np.float32)):
                    results[t][k] = results[t][k].tolist()
            if results[t]['score'] > Configs.camera_thresh:
                cam_obj.append([results[t]['class'], results[t]['tracking_id'],
                                results[t]['bbox'][0],
                                results[t]['bbox'][1],
                                results[t]['bbox'][2],
                                results[t]['bbox'][3]])

    return cam_obj


def lidar_json(scores, dt_box_lidar, types):
    lidar_obj = []
    if scores.size != 0:
        for i in range(scores.size):
            q = yaw2quaternion(float(dt_box_lidar[i][8]))
            orientation_x = q[1]
            orientation_y = q[2]
            orientation_z = q[3]
            orientation_w = q[0]
            position_x = float(dt_box_lidar[i][0])
            position_y = float(dt_box_lidar[i][1])
            position_z = float(dt_box_lidar[i][2])
            dimensions_l = float(dt_box_lidar[i][4])
            dimensions_w = float(dt_box_lidar[i][3])
            dimensions_h = float(dt_box_lidar[i][5])
            value = scores[i]
            label = int(types[i])
            if value > Configs.lidar_thresh:
                lidar_obj.append([label, position_x, position_y, position_z,
                                  dimensions_h, dimensions_w, dimensions_l,
                                  orientation_w])
    return lidar_obj


def lidar_to_cam(x, y, z, cam2lidar_calib):
    """
    :param x:
    :param y:
    :param z: 点云中3D点(x,y,z)
    :param cam2lidar_calib: 旋转矩阵(3x4)
    :return: 相机坐标系的点(x,y,z)
    """
    p = np.array([x, y, z, 1])
    p = np.matmul(cam2lidar_calib, p)
    p = p[0:3]  # cam coord
    return p


def roty(t):
    # Rotation about the y-axis.
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_velo_box_3D(x, y, z, h, w, l, R):
    """
    :param x:
    :param y:
    :param z:
    :param h:
    :param w:
    :param l:
    :param R: 点云中的航向角
    :return: 点云中3D_box的8个顶点
    """
    ry = -R - np.pi / 2
    rz = np.arctan2(math.sin(ry), math.cos(ry))
    r = roty(rz)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(r, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + x  ## 3D坐标的8个点
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    return np.transpose(corners_3d)


def velo3D_to_imgboxes(corners3d, cameraMatrix, img_shape_2d):
    """
    :param corners3d: 点云中的8个顶点
    :param cameraMatrix: 相机内参(3x4)
    :param img_shape_2d:RGB图像的尺寸 [1080, 1920, 3]
    :return:2D图像中的两个对角点
    """
    corner_3d = np.array(corners3d)
    corners3d_hom = np.concatenate((corner_3d, np.ones((8, 1))), axis=1)
    img_pts = np.matmul(corners3d_hom, cameraMatrix.T)
    x_p, y_p = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    x1, y1 = np.min(x_p), np.min(y_p)
    x2, y2 = np.max(x_p), np.max(x_p)
    img_boxes = [x1, y1, x2, y2]
    img_boxes[0] = np.clip(img_boxes[0], 0, img_shape_2d[1] - 1)
    img_boxes[1] = np.clip(img_boxes[1], 0, img_shape_2d[0] - 1)
    img_boxes[2] = np.clip(img_boxes[2], 0, img_shape_2d[1] - 1)
    img_boxes[3] = np.clip(img_boxes[3], 0, img_shape_2d[0] - 1)
    return img_boxes


def calculate_iou(box1, box2):
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1, 0) * \
            max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
    iou = 1.0 * inter / (area1 + area2 - inter)
    return iou


def calculate_id(cat_id):
    # class_name = ['other':0, 'car':1, 'truck':2, 'bus':3, 'trailer':4,
    #               'construction_vehicle':5,
    #               'pedestrian':6, 'motorcycle':7,
    #               'bicycle':8, 'traffic_cone':9, 'barrier':10]
    # if cat_id in ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle']:
    if cat_id in [1, 2, 3, 4, 5, 7]:
        obj_id = 1
        if cat_id == 1:
            obj_class = 10
        elif cat_id == 7:
            obj_class = 40
        elif cat_id == 2:
            obj_class = 30
        else:
            obj_class = 0
    elif cat_id == 8:
        obj_id = 2
        obj_class = 85
    elif cat_id == 6:
        obj_id = 3
        obj_class = 82
    else:
        obj_id = 0
        if cat_id == 9:
            obj_class = 50
        else:
            obj_class = 0
    return obj_id, obj_class


def get_distance_point(sensor_gps, sensor_roty, center_x, center_z):
    """
    :param sensor_gps: [longitude, latitude]
    :param sensor_roty:
    :param center_x:
    :param center_z:
    :return:
    """
    # direction: （north：0，east：90，south：180，west：270）
    start = geopy.Point(sensor_gps[1], sensor_gps[0])  # latitude, longitude, altitude
    d_x = distance(meters=center_x)
    p_x = d_x.destination(point=start, bearing=sensor_roty + 90)
    point = geopy.Point(p_x.latitude, p_x.longitude)  # latitude, longitude, altitude
    d_y = distance(meters=center_z)
    p = d_y.destination(point=point, bearing=sensor_roty)
    return p.longitude, p.latitude, p.altitude


def merge(lidar, img=None):
    """
    :param img: per img_obj: {"type":, 'id', 'x1', 'y1', 'x2', 'y2'}
    :param lidar: per lidar_obj: {"type":,"bbox": bbox,'x', 'y', 'z', 'h', 'w', 'l','R'}
    :return: participant: {"ptcType": obj_id,"ptcId": 11,"plateNo": 11,"pos": pos,"speed": 0,"heading": healding,"size": obj_size,"vehicleClass": obj_class}
    """
    if img is not None:
        cat_id = img["type"]
        ptcID = img["id"]
    else:
        cat_id = lidar["type"]
        ptcID = 0

    ptctype, vehicleclass = calculate_id(cat_id)
    longitude, latitude, altitude = get_distance_point(Configs.lidar_loc, Configs.lidar_roty, lidar["y"], lidar["x"])
    lon = round(longitude * 10000000)
    lat = round(latitude * 10000000)
    alt = round(altitude / 0.1)
    heading = round(math.degrees(lidar["r"]) / 0.0125)
    pos = {
        "longitude": lon,
        "latitude": lat,
        "altitude": alt
    }
    obj_size = {
        "width": round(lidar["w"] * 100),
        "length": round(lidar["l"] * 100),
        "height": round(lidar["h"] * 20)
    }

    ret = {
        "ptcType": ptctype,
        "ptcId": ptcID,
        "plateNo": "",
        "pos": pos,
        "speed": 0,
        "heading": heading,
        "size": obj_size,
        "vehicleClass": vehicleclass
    }
    return ret


def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()

    car_indices = get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices = get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices = get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices = get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices = get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices = get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices = get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices = get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices = get_annotations_indices(9, 0.1, label_preds_, scores_)

    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices +
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices +
                            motorcycle_indices
                            ])

    return img_filtered_annotations


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    return points


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg
