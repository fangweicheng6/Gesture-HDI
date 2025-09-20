import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from thop import profile
from al3d_utils import common_utils
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils

from al3d_det.models import fusion_modules
from .anchor_kitti import ANCHORKITTI
from al3d_det.utils import nms_utils
from al3d_det.models import image_modules as img_modules
from al3d_det.models import modules as cp_modules
from deepspeed.profiling.flops_profiler import FlopsProfiler

class ANCHORKITTIMM_LiDAR(ANCHORKITTI):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)

    def forward(self, batch_dict, cur_module=None, end=False):
        if not end:
            return cur_module(batch_dict)
        else:
            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()

                ret_dict = {
                    'loss': loss
                }

                return ret_dict, tb_dict, disp_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

class ANCHORKITTIMM_Camera(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.img_backbone = img_modules.__all__[model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=model_cfg.IMAGE_BACKBONE
        )
        if 'IMGPRETRAINED_MODEL' in model_cfg.IMAGE_BACKBONE and model_cfg.IMAGE_BACKBONE.IMGPRETRAINED_MODEL is not None:
            checkpoint= torch.load(model_cfg.IMAGE_BACKBONE.IMGPRETRAINED_MODEL, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            ckpt = state_dict
            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                if k.startswith('backbone'):
                    new_v = v
                    new_k = k.replace('backbone.', 'img_backbone.')
                else:
                    continue
                new_ckpt[new_k] = new_v
            self.img_backbone.load_state_dict(new_ckpt, strict=False)


class ANCHORMMKITTI(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.lidar = ANCHORKITTIMM_LiDAR(
            model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.camera = ANCHORKITTIMM_Camera(model_cfg)
        self.training = self.lidar.training
        self.second_stage = self.lidar.second_stage
        self.grid_size = self.lidar.dataset.grid_size[::-1] + [1, 0, 0]
        
        voxel_size  = self.lidar.dataset.voxel_size
        point_cloud_range = self.lidar.dataset.point_cloud_range
        self.freeze_img = model_cfg.IMAGE_BACKBONE.get('FREEZE_IMGBACKBONE', False)
        self.freeze()
        
    def freeze(self):
        if self.freeze_img:
            for param in self.camera.img_backbone.img_backbone.parameters():
                param.requires_grad = False

            for param in self.camera.img_backbone.neck.parameters():
                param.requires_grad = False
    def forward(self, batch_dict):
        
        # /detection/al3d_det/models/image_modules/mmdet_ffnkitti.py
        # 提取图像特征
        batch_dict = self.camera.img_backbone(batch_dict)
        
        # /detection/al3d_det/models/modules/backbone_3d/mean_vfe.py
        # MeanVFE 平均体素特征编码（Voxl feature encoding）用来提取每个体素内所有点的特征的平均值
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[0])
        
    # if self.training:  # 或根据你的需求设置调试条件
        # 计算 FLOPs 和总参数量
        # flops, total_params = profile(self, inputs=(batch_dict,))

        # # 单位转换（总参数量）
        # flops_in_gflops = flops / 1e9  # 转换为 GFLOPs
        # total_params_in_m = total_params / 1e6  # 转换为 M (百万)

        # # 计算并打印可训练的参数量
        # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # trainable_params_in_m = trainable_params / 1e6  # 转换为 M (百万)

        # # 打印结果
        # print(f"当前模型的FLOPs: {flops_in_gflops:.2f} GFLOPs")
        # print(f"当前模型的总参数量: {total_params_in_m:.2f} M")
        # print(f"当前模型的可训练参数量: {trainable_params_in_m:.2f} M")
        
        # 调用分析函数
        # print("剪枝前模型分析：")
        # analyze_model_deepspeed(self, batch_dict)
        
        
        # # 剪枝后分析
        # print("剪枝后模型分析：")
        # analyze_model_deepspeed(self, batch_dict)
        
        # /detection/al3d_det/models/modules/backbone_3d/backbone3d.py
        # 使用稀疏卷积提取点云多尺度的体素特征 得到输出encoded_spconv_tensor以及多尺度3d体素特征和步幅
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[1])
        
        # /detection/al3d_det/models/modules/backbone_2d/height_compression.py
        # 对encoded_spconv_tensor中的稀疏特征进行密集化 将深度D与通道C结合，形成新的特征图作为spatial_features
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[2])
        
        # /detection/al3d_det/models/modules/backbone_2d/backbone2d.py
        # 对输入的二维特征图spatial_features进行卷积得到多尺度特征图，再经过反卷积恢复尺度并沿着通道维度拼接在一起得到 spatial_features_2d
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[3])
        
        # /detection/al3d_det/models/modules/dense_heads/anchor_head_single.py
        # 得到一阶段类别和边界框的预测结果
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[4])
        
        
        if self.second_stage:
           
           # 
            batch_dict = self.lidar(batch_dict, self.lidar.module_list[5])
        # 计算损失    
        ret_lidar = self.lidar(batch_dict, end=True)

        return ret_lidar

    def update_global_step(self):
        if hasattr(self.lidar, 'update_global_step'):
            self.lidar.update_global_step()
        else:
            self.module.lidar.update_global_step()

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' %
                        checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' %
                            (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' %
                    (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(
                        optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(
                        optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' %
                  checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    
