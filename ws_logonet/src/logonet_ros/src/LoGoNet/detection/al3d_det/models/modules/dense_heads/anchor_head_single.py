import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate
from al3d_det.models.modules.roi_heads.LoGoHead_kitti import LoGoHeadKITTI
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D
import matplotlib.pyplot as plt
import cv2
import os
import time

class GateFusion(nn.Module):
    def __init__(self, in_channels):
        # 这句代码是pyhton中的继承机制；super方法用于子类调用父类
        super(GateFusion, self).__init__()
    
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, 128, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size = 1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, features_1, features_2):
        combined_features = torch.cat([features_1, features_2],dim=1) # [B, C1 + C2, H, W]
        gate = self.gate_conv(combined_features)

        fused_features = gate * features_1 + (1 - gate) * features_2

        return fused_features


# 定义自注意力层
class SelfAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
    
    def forward(self, x):   
        # x.shape: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # 将 [B, C, H, W] 展开成 [B, H*W, C]
        
        # 使用自注意力机制进行融合
        attn_output, _ = self.attn(x_flat, x_flat, x_flat)  # 输入、键和值都是相同的
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # 恢复形状 [B, C, H, W]

        return attn_output

class BEVTransformer(nn.Module):
    def __init__(self, bev_h, bev_w, in_channels, num_heads=8, num_layers=6):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # BEV Query 初始化: 可学习的查询向量
        self.bev_query = nn.Parameter(torch.randn(bev_h * bev_w, in_channels))  # [H*W, C]

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, bev_h * bev_w, in_channels))  # [1, H*W, C]

        # Transformer encoder
        self.attn_layer = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

        # BEV特征恢复卷积层
        self.bev_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image_features):
        """
        :param image_features: [B, C, H_img, W_img]
        :return: bev_feats: [B, C, H_bev, W_bev]
        """
        B, C, H_img, W_img = image_features.shape

        image_flat = image_features.view(B, C, -1).permute(0, 2, 1)  # [B, H_img*W_img, C]

        # 创建 BEV Query 和位置编码
        bev_query = self.bev_query.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, C]
        
        # 位置编码，增加空间感知：将位置编码加到 BEV Query 上
        bev_query = bev_query + self.positional_encoding

        # 使用 Transformer 进行交叉注意力
        bev_feat, _ = self.attn_layer(query=bev_query, key=image_flat, value=image_flat)  # [B, H*W, C]

        # 恢复 BEV 特征到 [B, C, H_bev, W_bev] 形状
        bev_feat = bev_feat.view(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2)  # [B, C, H_bev, W_bev]

        # BEV 特征卷积处理，进一步提取 BEV 维度特征
        bev_feat = self.bev_conv(bev_feat)  # [B, C, H_bev, W_bev]

        return bev_feat

class BEVCrossAttention(nn.Module):
    def __init__(self, bev_h, bev_w, in_channels):
        super().__init__()
        self.bev_query = nn.Parameter(torch.randn(bev_h * bev_w, in_channels))  # 可学习的BEV Query
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, image_features):
        # image_features: [B, C, H_img, W_img]
        B, C, H, W = image_features.shape
        image_flat = image_features.view(B, C, -1).permute(0, 2, 1)  # [B, H_img*W_img, C]
        bev_query = self.bev_query.unsqueeze(0).expand(B, -1, -1)     # [B, H_bev*W_bev, C]
        
        # 交叉注意力：BEV Query作为Query，图像特征作为Key/Value
        bev_feat, _ = self.attn(bev_query, image_flat, image_flat)
        return bev_feat.view(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2)  # [B, C, H_bev, W_bev]


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.pe = PositionalEncoding2D(input_channels)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        self.image_down_sample = nn.Sequential(
            nn.Conv2d(256, input_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.bev_img = BEVCrossAttention(bev_h=10, bev_w=34, in_channels = input_channels)

        # 使用 BEVTransformer 来映射图像特征到 BEV 空间
        # self.bev_img = BEVTransformer(bev_h=10, bev_w=34, in_channels=input_channels)

        self.gate_fusion = GateFusion(in_channels = input_channels)

        self.self_attention = SelfAttentionFusion(in_channels = input_channels*2)

        # 特征融合模块
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_channels*2, input_channels, 3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        # 在初始化中添加独立分支
        self.cls_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )

        self.reg_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        for m in self.fusion_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



    def forward(self, data_dict):
        
        spatial_features_2d = data_dict['spatial_features_2d']
        # 提取各个维度的值
        batch_size, channels, h, w = spatial_features_2d.shape
        image_feature = data_dict['image_features']['layer1_feat2d']

        # BCHW -> BHWC
        spatial_features_2d = spatial_features_2d.permute(0, 2, 3, 1)
        # 添加位置编码 BHWC
        spatial_features_2d += self.pe(spatial_features_2d)
        # BHWC -> BCHW
        spatial_features_2d = spatial_features_2d.permute(0, 3, 1, 2) # [10,34]

        image_feat = self.image_down_sample(image_feature)  # [B, C, H, W]

        # 对齐空间尺寸
        # if image_feat.shape[-2:] != spatial_features_2d.shape[-2:]:
        #     image_feat = F.interpolate(
        #         image_feat, 
        #         size=spatial_features_2d.shape[-2:], 
        #         mode='bilinear', 
        #         align_corners=False
        #     )
        # bev_image_feat = image_feat

        # 将图像映射到bev空间与点云对齐
        bev_image_feat = self.bev_img(image_feat)
        # 在 BEVCrossAttention 后添加尺寸检查
        assert bev_image_feat.shape[-2:] == spatial_features_2d.shape[-2:], \
            f"BEV 特征尺寸不匹配: {bev_image_feat.shape} vs {spatial_features_2d.shape}"
        
        # 拼接图像特征和空间特征
        cat_features = torch.cat((bev_image_feat, spatial_features_2d), dim=1)  # [B, 2*C, H, W]

        # 门控融合两种模态的原始特征
        gated_features = self.gate_fusion(bev_image_feat, spatial_features_2d)
        
        fused_features = self.fusion_conv(cat_features)

        # 自注意力机制丰富特征
        final_feats = self.self_attention(fused_features) + gated_features  # 残差连接

        final_feats = fused_features + gated_features  # 残差连接
        
        # 简化融合流程
        # fused_features = torch.cat([bev_image_feat, spatial_features_2d], dim=1)
        # final_feats = self.fusion_conv(fused_features) + spatial_features_2d  # 残差连接

        
        # 在forward中分开处理解耦
        cls_feat = self.cls_conv(final_feats)
        reg_feat = self.reg_conv(final_feats)

        cls_preds = self.conv_cls(cls_feat)
        box_preds = self.conv_box(reg_feat)

        
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training: 
            
            # 模型预测的偏移量解码为相对于锚框的实际边界框坐标
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
