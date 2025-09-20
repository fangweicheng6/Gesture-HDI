import torch
import cv2
import numpy as np
from pathlib import Path
from LoGoNet.detection.al3d_det.models import build_network, load_data_to_gpukitti
from LoGoNet.detection.al3d_det.datasets.processor.point_feature_encoder import PointFeatureEncoder
from LoGoNet.detection.al3d_det.datasets.processor.data_processor import DataProcessor
from LoGoNet.utils.al3d_utils.config_utils import cfg_from_yaml_file, cfg

from LoGoNet.detection.predict_dir.predict import DataBuilder
from LoGoNet.detection.predict_dir.eval_utils import build_batch_dict

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


class LoGoNet:
    def __init__(self, config_file, model_path):
        cfg_from_yaml_file(config_file, cfg)
        self.cfg = cfg
        self.predict_set = DataBuilder(dataset_cfg=self.cfg['DATA_CONFIG'],
        class_names=self.cfg['CLASS_NAMES'],
        training=False,
        root_path=None,
        logger=logger)
        
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process(self, image_path, point_cloud_path):
        
        image_data = load_image(image_path)
        pc_data = load_point_cloud(point_cloud_path)

        batch_dict = build_batch_dict(image_data, pc_data, self.predict_set)
        
        load_data_to_gpukitti(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = self.model(batch_dict)
        return pred_dicts