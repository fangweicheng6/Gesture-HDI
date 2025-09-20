import os
import re
import glob
import time
import argparse
import datetime
from pathlib import Path

import cv2
from torch.utils.data import DataLoader

import numpy as np
import torch
from tensorboardX import SummaryWriter

from al3d_utils import common_utils
from al3d_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from al3d_utils.model_utils import load_params_from_file, load_params_with_optimizer

from al3d_det.datasets import build_dataloader
from al3d_det.models import build_network
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前文件所在目录添加到sys.path
sys.path.append(current_dir)
from predict_dir import eval_utils

from al3d_det.datasets.augmentor.data_augmentor import DataAugmentor
from al3d_det.datasets.processor.data_processor import DataProcessor
from al3d_det.datasets.processor.point_feature_encoder import PointFeatureEncoder
from al3d_det.datasets.augmentor.test_time_augmentor import TestTimeAugmentor


def parse_config(cfg_file, ckpt_file):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_file, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=ckpt_file, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def load_image(image_path):
    image_file = Path(image_path)
    assert image_file.exists()
    image = cv2.imread(str(image_file))
    image = image.astype(np.float32)
    image /= 255.0
    return image


def load_point_cloud(point_cloud_path):
    pc_file = Path(point_cloud_path)
    assert pc_file.exists()
    return np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 4)


class DataBuilder:
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):

        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        self.use_image = getattr(self.dataset_cfg, "USE_IMAGE", False)
        self.image_scale = getattr(self.dataset_cfg, "IMAGE_SCALE", 1)
        self.load_multi_images = getattr(self.dataset_cfg, "LOAD_MULTI_IMAGES", False)
        self.ceph = False
        if self.dataset_cfg is None or class_names is None:
            return
        if getattr(self.dataset_cfg, 'OSS_PATH', None) is not None:
            self.root_path = self.dataset_cfg.OSS_PATH
            self.ceph = True
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, self.use_image, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None
        self.kitti_infos = []
        self.include_kitti_data()

    def include_kitti_data(self):
        kitti_infos = []

        # 定义四个字典
        point_cloud = {
            'num_features': 4,
            'lidar_idx': '000001'
        }

        image = {
            'image_idx': '000001',
            'image_shape': np.array([376, 672], dtype=np.int32)
        }

        calib = {
            'P2': np.array([[264.0, 0.0, 343.76000977, 0.0],
                            [0.0, 263.70001221, 183.87950134, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),

            'R0_rect': np.array([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),

            'Tr_velo_to_cam': np.array([[0.0293523, -0.99911702, 0.0300603, 0.108313],
                                        [-0.0142596, -0.0304887, -0.99943298, 0.0907808],
                                        [0.99946702, 0.028907, -0.015142, 0.384258],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        }

        # 将这些字典添加到一个新的字典中
        new_entry = {
            'point_cloud': point_cloud,
            'image': image,
            'calib': calib
        }

        # 添加新的字典到kitti_infos
        kitti_infos.append(new_entry)
        self.kitti_infos.extend(kitti_infos)


def build_data(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
               logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, length=0):
    # 实例化DataBuilder并返回相关数据
    dataset = DataBuilder(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=training,
        root_path=root_path,
        logger=logger
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    sampler = None

    return dataset, sampler


def load_model(cfg_file, ckpt_file):
    args, cfg = parse_config(cfg_file, ckpt_file)

    dist_test = False
    # total_gpus = 1

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'


    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    # logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    # logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # if dist_test:
    #     logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    # for key, val in vars(args).items():
    #     logger.info('{:16} {}'.format(key, val))
    # log_config_to_file(cfg, logger=logger)

    predict_set, sampler = build_data(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=predict_set)
    load_params_from_file(model, filename=args.ckpt, logger=logger, to_cpu=False, fix_pretrained_weights=False)
    model.cuda()
    model.eval()

    # 将相关参数存储到cfg中
    cfg.PREDICT_SET = predict_set
    cfg.EPOCH_ID = epoch_id
    cfg.LOGGER = logger
    cfg.EVAL_OUTPUT_DIR = eval_output_dir
    cfg.SAVE_TO_FILE = args.save_to_file

    return model, cfg


def predict(model, image_data, pc_data, cfg):
    with torch.no_grad():
        # start evaluation
        image_data, pc_data, annos = eval_utils.eval_one_epoch(
            image_data, pc_data, cfg.PREDICT_SET, cfg, model, cfg.EPOCH_ID, cfg.LOGGER,
            result_dir=cfg.EVAL_OUTPUT_DIR, save_to_file=cfg.SAVE_TO_FILE
        )
        return image_data, pc_data, annos
