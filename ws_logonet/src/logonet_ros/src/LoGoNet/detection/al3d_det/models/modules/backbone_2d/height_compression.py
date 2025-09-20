import torch.nn as nn

# DHW(zdim, ydim, xdim)
class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if 'fusion_feature' not in batch_dict and 'encoded_spconv_tensorlist' not in batch_dict:
            
            # 提取 encoded_spconv_tensor 并将其转换为稠密张量
            # 稀疏张量是指只存储非零元素 稀疏数据结构可以显著提高计算效率和节省内存
            # 稠密张量是指也存储零元素是完整的 常用于常规的机器学习任务和图像处理
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
            
        elif 'encoded_spconv_tensorlist' in batch_dict:
            spatial_features = batch_dict['encoded_spconv_tensorlist'][0].dense()
            N, C, D, H, W = spatial_features.shape
            
            spatial_features = spatial_features.reshape(N, C * D, H, W)
            for each in batch_dict['encoded_spconv_tensorlist'][1:]:
                each = each.dense()
                N, C, D, H, W = each.shape
                each = each.reshape(N, C * D, H, W)
                spatial_features = spatial_features + each
            batch_dict['spatial_features'] = spatial_features
            batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
            return batch_dict
        else:
            spatial_features = batch_dict['fusion_feature']
            
        # 2,128,2,40,34
        N, C, D, H, W = spatial_features.shape
        
        # 将深度维度 D 与通道维度 C 结合，形成新的特征图
        # 这里是对体素特征在z轴方向堆叠，也就是压缩
        spatial_features = spatial_features.reshape(N, C * D, H, W)
        
        
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        
        return batch_dict
