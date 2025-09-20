import torch
import torch.nn as nn
import numpy as np

class LearnablePositionEncoding(nn.Module):
    def __init__(self, channels, max_shape=(200, 200)):
        super().__init__()
        self.height_encoder = nn.Parameter(torch.randn(1, channels, max_shape[0], 1))
        self.width_encoder = nn.Parameter(torch.randn(1, channels, 1, max_shape[1]))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        h_enc = self.height_encoder[:, :, :H, :]  # [1, C, H, 1]
        w_enc = self.width_encoder[:, :, :, :W]   # [1, C, 1, W]
        return x + h_enc + w_enc
    

class SinePositionEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 计算 div_term
        self.div_term = torch.exp(torch.arange(0, channels, 2).float() * (-np.log(10000.0) / channels))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device
        
        # 生成坐标网格
        y_pos = torch.arange(H, device=device).float().unsqueeze(1)  # [H, 1]
        x_pos = torch.arange(W, device=device).float().unsqueeze(0)  # [1, W]
        
        # 初始化位置编码
        pe = torch.zeros(1, self.channels, H, W, device=device)
        
        # 计算正弦和余弦编码
        pe[0, 0::2] = torch.sin(y_pos * self.div_term.unsqueeze(1))  # 对 y 方向编码
        pe[0, 1::2] = torch.cos(y_pos * self.div_term.unsqueeze(1))  # 对 y 方向编码
        
        pe[0, 0::2] += torch.sin(x_pos * self.div_term.unsqueeze(0))  # 对 x 方向编码
        pe[0, 1::2] += torch.cos(x_pos * self.div_term.unsqueeze(0))  # 对 x 方向编码
        
        # 将位置编码加到输入上
        return x + pe