import Configs
import numpy as np
import torch

from ret2json import remove_low_score_nu
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d import __version__, torchie

import src._init_paths
import os
from opts import opts
from detector import Detector


class Lidar(object):
    def __init__(self):
        self.points = None
        self.config_path = Configs.lidar_config_file_path
        self.model_path = Configs.lidar_model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.read_config()

    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )
        print("#########  CenterPoint Init Success  #########")

    def run(self, points):
        # print(f"input points shape: {points.shape}")
        num_features = 5
        self.points = points.reshape([-1, num_features])
        self.points[:, 4] = 0  # timestamp value

        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]
        )
        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]
        # print(f"output: {outputs}")
        outputs = remove_low_score_nu(outputs, 0.45)
        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()
        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        return scores, boxes_lidar, types


class Camera(object):
    def __init__(self):
        opt = opts().init()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.load_model = Configs.camera_model_path
        self.detector = Detector(opt)
        print("#########  CenterTrack Init Success  #########")
        self.input_meta = {'pre_dets': [], 'calib': Configs.camera_calib}
