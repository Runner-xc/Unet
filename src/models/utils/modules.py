import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from mamba_ssm import Mamba


"""-------------------------------------------------Convolution----------------------------------------------"""
import torch
from torchvision.ops import DeformConv2d

class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        # 动态计算偏移量通道数
        offset_channels = 2 * k * k
        self.offset_conv = nn.Conv2d(in_channels, offset_channels, kernel_size=k, padding=k//2)
        self.deform_conv = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=k, 
            padding=k//2,
            bias=False # BatchNorm会处理偏置，这里设为False
        )

    def forward(self, x):
        offsets = self.offset_conv(x)
        return self.deform_conv(x, offsets)

class DWConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, 
            padding=kernel_size // 2, 
            groups=in_channels, bias=False
        )
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class Att_DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel=None, kernel_size=3):
        super(Att_DWConv, self).__init__()
        if mid_channel is None and in_channels != 3:
            mid_channel = in_channels // 2
        else: 
            mid_channel = in_channels
        self.dconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels)
        self.pconv1 = nn.Conv2d(in_channels, mid_channel, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid_channel)

        self.dconv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=kernel_size, padding=kernel_size//2, groups=mid_channel)
        self.pconv2 = nn.Conv2d(mid_channel, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SE_Block(out_channels)
        

    def forward(self, x):
        x = self.bn1(self.relu(self.pconv1(self.dconv1(x))))
        return self.bn2(self.relu(self.pconv2(self.dconv2(x))))
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels // 4
        # 初始定义分离的各层
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(mid_channels), 
            self.relu)
        
        self.cbr2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            self.relu)
        
    def forward(self, x):
        x = self.cbr1(x)  
        return self.cbr2(x)

class Axis_wise_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(Axis_wise_Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel), padding=(0, kernel // 2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(kernel, 1), padding=(kernel // 2, 0), groups=in_channels))
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pwconv(self.conv(x))

class AWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(AWConv, self).__init__()
        self.conv = nn.Sequential(
            Axis_wise_Conv2d(in_channels, out_channels, kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        
    def forward(self, x):
        return self.conv(x)
    
class Att_AWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(Att_AWConv, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            Axis_wise_Conv2d(in_channels, out_channels, kernel),
            nn.BatchNorm2d(out_channels),
            self.relu)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            self.relu,
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1))
        self.sgm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        weight = self.sgm(self.fc(self.gap(x)))
        return x * weight
                    
class Att_AWBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.awconv5x5 = AWConv(in_channels, out_channels, kernel=5)
        self.se = SE_Block(out_channels)
        
    def forward(self, x):
        return self.se(self.awconv5x5(x))
    
class DWDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels // 4
        # 调整顺序为 DWConv → BN → ReLU
        self.relu = nn.ReLU(inplace=True)
        
        self.dconv1 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            self.relu
        )
        self.dconv2 = nn.Sequential(
            DWConv(mid_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            self.relu
        )   

    def forward(self, x):
        return self.dconv2(self.dconv1(x))
    
class Conv_3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Conv_3, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 4
        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))

        self.c2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(mid_channels), 
                                nn.ReLU(inplace=True))

        self.c3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(out_channels), 
                                nn.ReLU(inplace=True))
        

    def forward(self, x1):
        return self.c3(self.c2(self.cbr1(x1)))
    

class Dalit_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Dalit_Conv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3), 
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
        

    def forward(self, x):
        return self.cbr3(self.cbr2(self.cbr1(x)))
    
class DWDalit_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DWDalit_Conv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        self.cbr1 = nn.Sequential(DWConv(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(DWConv(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(DWConv(mid_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3), 
                                nn.BatchNorm2d(out_channels), 
                                nn.ReLU(inplace=True))
        

    def forward(self, x):
        return self.cbr3(self.cbr2(self.cbr1(x)))
    
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConv, self).__init__()

        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), 
                                nn.BatchNorm2d(out_channels)) 
              
        self.cbr2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(out_channels), 
                                  nn.ReLU(inplace=True))
        
        self.c3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(out_channels))
                
    def forward(self, x):
        residual = x
        re = self.c1(residual)
        x = self.cbr2(x)
        x = self.c3(x)
        x = x + re
        return self.relu(x)
    
class ResDConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResDConv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3), 
                                  nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        residual = x
        re = self.c1(residual)
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = x + re
        return self.relu(x)
    
class DWResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWResConv, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), 
                               nn.BatchNorm2d(out_channels))
        
        self.cbr2 = nn.Sequential(DWConv(in_channels, out_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(out_channels), 
                                  nn.ReLU(inplace=True))
        self.cbr3 = nn.Sequential(DWConv(out_channels, out_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(out_channels))
              
    def forward(self, x):
        residual = x
        re = self.cbr1(residual)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = x + re
        return self.relu(x)
class DWResDConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DWResDConv, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

        self.cbr1 = nn.Sequential(DWConv(in_channels, mid_channels, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr2 = nn.Sequential(DWConv(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2), 
                                  nn.BatchNorm2d(mid_channels), 
                                  nn.ReLU(inplace=True))
        
        self.cbr3 = nn.Sequential(DWConv(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3), 
                                  nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        residual = x
        re = self.c1(residual)
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = x + re
        return self.relu(x)
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,      # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width 1×1
                expand=expand,    # Block expansion factor
        )
        
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class SCS_Mamba(nn.Module):
    """
    选择性通道-空间Mamba (Selective Channel-Space Mamba) 模块.
    
    该模块包含两个并行的Mamba路径:
    1. 空间选择路径 (Spatial Selection Path): 学习空间位置之间的依赖关系。
       - 操作流程: (B, C, H, W) -> (B, H*W, C) -> Mamba -> (B, H*W, C)
    2. 通道选择路径 (Channel Selection Path): 学习特征通道之间的依赖关系。
       - 操作流程: (B, C, H, W) -> (B*H*W, C, 1) -> Mamba -> (B*H*W, C, 1)

    两条路径的输出通过可学习的门控权重进行融合，并与原始输入相加（残差连接）。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        """
        初始化模块.
        
        参数:
            dim (int): 输入特征图的通道数 (C).
            d_state (int): Mamba的SSM状态扩展因子.
            d_conv (int): Mamba的局部卷积宽度.
            expand (int): Mamba块的扩展因子.
        """
        super().__init__()
        self.dim = dim
        
        # --- 空间选择路径的组件 ---
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_mamba = Mamba(
            d_model=dim, 
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # --- 通道选择路径的组件 ---
        # 对于通道路径，我们将每个空间位置的C个通道视为一个序列
        # Mamba的d_model将是1，因为它处理的是每个通道的单个值
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mamba = Mamba(
            d_model=1,  # 核心区别：在通道维度上扫描
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # --- 融合门控 ---
        # 使用两个可学习的标量参数作为门控权重，以平衡两条路径的贡献
        # 初始化为1，让模型从均等贡献开始学习
        self.gate_s = nn.Parameter(torch.ones(1))
        self.gate_c = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        前向传播.
        
        参数:
            x (Tensor): 输入特征图，形状为 (B, C, H, W).
        返回:
            Tensor: 输出特征图，形状与输入相同.
        """
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C, H, W = x.shape
        assert C == self.dim, f"输入通道数 {C} 与模块维度 {self.dim} 不匹配"

        # --- 空间选择路径 ---
        x_spatial_flat = x.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        x_spatial_norm = self.spatial_norm(x_spatial_flat)
        spatial_out = self.spatial_mamba(x_spatial_norm)         # (B, H*W, C)
        
        # --- 通道选择路径 ---
        x_channel_permuted = x.permute(0, 2, 3, 1)               # (B, H, W, C)
        x_channel_flat = x_channel_permuted.reshape(B * H * W, C) # (B*H*W, C)
        
        x_channel_norm = self.channel_norm(x_channel_flat)      # 在C维度上归一化
        channel_out_flat = self.channel_mamba(x_channel_norm.unsqueeze(-1)) # (B*H*W, C, 1)
        channel_out = channel_out_flat.squeeze(-1).reshape(B, H, W, C) # (B, H, W, C)
        
        # ---融合与残差连接 ---
        # 将空间路径的输出也恢复为 (B, H, W, C) 以便融合
        spatial_out_permuted = spatial_out.reshape(B, H, W, C)
        fused_out = (self.gate_s * spatial_out_permuted + 
                     self.gate_c * channel_out)
        
        fused_out_restored = fused_out.permute(0, 3, 1, 2)
        output = x + fused_out_restored
        
        return output
    
"""---------------------------------------------Pyramid Pooling Module----------------------------------------------------"""
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播，直接返回大于0的输入作为1，否则为0
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播，使用hardtanh函数来处理梯度
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        return STEFunction.apply(x)
    
# 因果图块（Causality Map Block）
class CausalityMapBlock(nn.Module):
    def __init__(self):
        super(CausalityMapBlock, self).__init__()
        # print("CausalityMapBlock initialized")

    def forward(self, x):
        """
        计算因果图，输入x的形状为(bs, k, n, n)，其中bs是批量大小，k是特征图的数量，
        n是特征图的空间维度。
        
        :param x: 网络的潜在特征
        :return: 因果图，形状为(bs, k, k)
        """
        if torch.isnan(x).any():
            # print("...the current feature maps object contains NaN")
            raise ValueError

        # 计算每个特征图的最大值
        maximum_values = torch.max(torch.flatten(x, 2), dim=2)[0]
        MAX_F = torch.max(maximum_values, dim=1)[0]
        x_div_max = x / (MAX_F.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-8)

        x = torch.nan_to_num(x_div_max, nan=0.0)

        # 计算因果图
        sum_values = torch.sum(torch.flatten(x, 2), dim=2)
        sum_values = torch.nan_to_num(sum_values, nan=0.0)
        maximum_values = torch.max(torch.flatten(x, 2), dim=2)[0]
        mtrx = torch.einsum('bi,bj->bij', maximum_values, maximum_values)
        tmp = mtrx / (sum_values.unsqueeze(1) + 1e-8)
        causality_maps = torch.nan_to_num(tmp, nan=0.0)

        # 归一化因果图
        max_cmaps = torch.max(causality_maps, dim=1, keepdim=True)[0]
        min_cmaps = torch.min(causality_maps, dim=1, keepdim=True)[0]
        causality_maps = (causality_maps - min_cmaps) / (max_cmaps - min_cmaps + 1e-8)

        # print(f"causality_maps: min {torch.min(causality_maps)}, max {torch.max(causality_maps)}, mean {torch.mean(causality_maps)}")
        return causality_maps


# 因果因子提取器（Causality Factors Extractor）
class CausalityFactorsExtractor(nn.Module):
    def __init__(self):
        super(CausalityFactorsExtractor, self).__init__()
        self.STE = StraightThroughEstimator()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # print("CausalityFactorsExtractor initialized")

    def forward(self, x, causality_maps):
        """
        根据因果图计算因果因子，输入x的形状为(bs, k, h, w)，其中bs是批量大小，k是特征图的数量，
        h和w是特征图的空间维度。因果图的形状为(bs, k, k)。
        
        :param x: 网络的潜在特征
        :param causality_maps: 因果图
        :return: 因果加权的特征图
        """
        triu = torch.triu(causality_maps, 1)
        tril = torch.tril(causality_maps, -1).permute((0, 2, 1)).contiguous()

        e = tril - triu
        e = self.STE(e)
        e = e.permute((0, 2, 1))

        f = triu - tril
        f = self.STE(f)
        bool_matrix = e + f

        by_col = torch.sum(bool_matrix, 2)
        by_row = torch.sum(bool_matrix, 1)

        multiplicative_factors = by_col - by_row
        multiplicative_factors = self.relu(multiplicative_factors)
        max_factors = torch.max(multiplicative_factors, dim=1, keepdim=True)[0]
        min_factors = torch.min(multiplicative_factors, dim=1, keepdim=True)[0]
        multiplicative_factors = (multiplicative_factors - min_factors) / (max_factors - min_factors + 1e-8)

        # print(f"multiplicative_factors: min {torch.min(multiplicative_factors)}, max {torch.max(multiplicative_factors)}, mean {torch.mean(multiplicative_factors)}")
        return torch.einsum('bkmn,bk->bkmn', x, multiplicative_factors)
class PSPModule(nn.Module):
    def __init__(self, in_channels: int, bin_size_list: list = [1, 2, 4, 8]):
        super(PSPModule, self).__init__()
        branch_channels = in_channels // 4        ## C/4
        self.branches = nn.ModuleList()
        # CAB 模块
        self.causality_map_block = CausalityMapBlock()
        self.causality_factors_extractor = CausalityFactorsExtractor()
        for i in range(len(bin_size_list)):
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin_size_list[i]),  # 使用平均池化
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU()
            )
            self.branches.append(branch)       

    def forward(self, inputs):
        if not self.branches:
            return inputs
        final = None
        for i, branch in enumerate(self.branches):
            out = branch(inputs)
            # # CAB 模块
            # causality_maps = self.causality_map_block(out)
            # enhanced_features = self.causality_factors_extractor(out, causality_maps)
            
            out = F.interpolate(out, size=inputs.shape[2:], mode='bilinear', align_corners=True)
            if final is None:
                final = out
            else:
                final = torch.cat([final, out], dim=1)
        final = torch.cat([inputs, final], dim=1)  # 将各特征图在通道维上拼接起来
        return final

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features
class DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DenseASPPBlock, self).__init__()
        self.aspp_1 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 1, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_2 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 2, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_3 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 3, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_5 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 5, 0.1,
                                      norm_layer, norm_kwargs)
        # self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                    #   norm_layer, norm_kwargs)
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 4, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        aspp1 = self.aspp_1(x)
        x = torch.cat([aspp1, x], dim=1)

        aspp2 = self.aspp_2(x)
        x = torch.cat([aspp2, x], dim=1)

        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp5 = self.aspp_5(x)
        x = torch.cat([aspp5, x], dim=1)

        # aspp24 = self.aspp_24(x)
        # x = torch.cat([aspp24, x], dim=1)

        x = self.cbr1(x)
        return x
    
"""---------------------------------------------AMSFN----------------------------------------------------""" 
class AMSFN(nn.Module):  #Adaptive Convolutional Pooling Network (ACPN)
    def __init__(self, in_channels, out_channels=None, mid_channels=None,):
        super(AMSFN, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.conv1x1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))

        # 3×3
        self.cbr1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbr2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbr3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sgm = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        

    def forward(self, x):
        b, c, h, w = x.size()
        c1 = self.conv1x1(x)
        s1 = self.avg_pool(c1)

        c2 = self.cbr1(x)
        s2 = self.avg_pool(c2)

        c3 = self.cbr2(x)
        s3 = self.avg_pool(c3)

        c4 = self.cbr3(x)
        s4 = self.avg_pool(c4)

        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.mlp(out)
        weights = self.sgm(out)

        # 将权重作用到各个卷积结果上并相加
        c1_weighted = c1 * weights[:, 0:c   , ...]
        c2_weighted = c2 * weights[:, c:c*2 , ...]
        c3_weighted = c3 * weights[:, c*2:c*3, ...]
        c4_weighted = c4 * weights[:, c*3:  , ...]

        output = c1_weighted + c2_weighted + c3_weighted + c4_weighted
        output = self.final_conv(output)
        return output
    
class AMSFNV2(AMSFN):  #Adaptive Convolutional Pooling Network (ACPN)
    def __init__(self, in_channels, out_channels=None, mid_channels=None,):
        super(AMSFNV2, self).__init__(in_channels, 
                                      out_channels, 
                                      mid_channels)
        if out_channels is None:
            out_channels = in_channels
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.conv1x1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))

        # 3×3
        self.cbr1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbr2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=5),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbr3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=7),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return super(AMSFNV2, self).forward(x)

class SpatialChannelAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        if in_ch % 4 != 0:
            out_ch = 1
        else:
            out_ch = in_ch // 4
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch, in_ch, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_ch, 1, 3, padding=1),
            nn.Sigmoid()
        )
          
    def forward(self, x):
        channel_weight = self.channel_att(x)  # [B,C,1,1]
        spatial_weight = self.spatial_att(x)  # [B,1,H,W]
        return x * channel_weight * spatial_weight  # 双重注意力
class AMSFNV3(nn.Module):
    def __init__(self, in_channels, out_channels=None, k=5):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        # 分支1: 1x1卷积保留细节
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            SpatialChannelAttention(in_channels)
        )
        
        # 分支2: 轴向卷积
        self.branch2 = AWConv(in_channels, in_channels, kernel=k)
        
        # 分支3: 可变形卷积适应形状
        self.branch3 = nn.Sequential(
            DeformConvBlock(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 动态融合权重生成
        self.fusion_weights = nn.Conv2d(in_channels*3, 3, 3, padding=1)  # 空间敏感权重
        
        # 输出变换
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        

    def forward(self, x):
        # 多流并行计算 提高训练速度
        s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        
        with torch.cuda.stream(s1):
            b1 = self.branch1(x)
        with torch.cuda.stream(s2):
            b2 = self.branch2(x)
        with torch.cuda.stream(s3):
            b3 = self.branch3(x)
        torch.cuda.synchronize()

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # 生成空间注意力权重
        weight_map = torch.softmax(self.fusion_weights(torch.cat([b1, b2, b3], dim=1)), dim=1)
        w1, w2, w3 = weight_map.chunk(3, dim=1)
        
        # 加权融合
        fused = w1*b1 + w2*b2 + w3*b3
        return self.final(fused)

class AMSFNV4(nn.Module):
    def __init__(self, in_channels, out_channels=None, k=5):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        # 分支1: 1x1卷积保留细节
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            SpatialChannelAttention(in_channels)
        )

        # 分支2: 轴向卷积
        self.branch2 = Att_AWConv(in_channels, in_channels, kernel=k)

        # 分支3: 可变形卷积适应形状
        self.branch3 = nn.Sequential(
            DeformConvBlock(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())

        # 动态融合权重生成
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(in_channels*3,3,3, padding=1, groups=3),
            nn.Conv2d(3,3,1)
        )
        
        # 输出变换
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        # 多流并行计算 提高训练速度
        s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        
        with torch.cuda.stream(s1):
            b1 = self.branch1(x)
        with torch.cuda.stream(s2):
            b2 = self.branch2(x)
        with torch.cuda.stream(s3):
            b3 = self.branch3(x)
        torch.cuda.synchronize()

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # 生成空间注意力权重
        weight_map = torch.softmax(self.fusion_weights(torch.cat([b1, b2, b3], dim=1)), dim=1)
        w1, w2, w3 = weight_map.chunk(3, dim=1)
        
        # 加权融合
        fused = w1*b1 + w2*b2 + w3*b3
        return self.final(fused)

class MSFN_Rep(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_repped = False

        # --- 所有分支都是可融合的 ---
        self.dwconv5x5 = nn.Sequential(DWConv(in_channels, out_channels, 5, 1), nn.BatchNorm2d(out_channels))
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.conv3x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(out_channels))
        self.conv1x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(out_channels))
        # 保持分支数量，增加一个额外的1x1分支 (可选，但可以增加模型的复杂度)
        self.conv1x1_extra = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        
        # 推理时融合后的单一卷积
        self.reparam_conv = None

    def forward(self, x):
        if hasattr(self, 'reparam_conv') and self.reparam_conv is not None:
            # 推理模式：执行单一的、融合后的 5x5 卷积
            return self.reparam_conv(x)
        else:
            # 训练模式：将所有分支的输出相加
            s1 = self.dwconv5x5(x)
            s2 = self.conv1x1(x) + self.conv3x1(x) + self.conv1x3(x)
            s3 = self.conv1x1_extra(x)
            return s1 + s2 + s3

    def eval(self):
        if not self.is_repped: self._auto_switch_to_deploy()
        return super().eval()

    def _auto_switch_to_deploy(self):
        if self.is_repped: return
        
        # 融合所有分支
        kernel, bias = self.get_equivalent_kernel_bias()
        
        # 创建等效的单一 5x5 卷积层
        self.reparam_conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=5, padding=2, bias=True,
            device=kernel.device, dtype=kernel.dtype
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        
        # 删除所有原有分支，释放内存
        self.__delattr__('dwconv5x5')
        self.__delattr__('conv1x1')
        self.__delattr__('conv3x1')
        self.__delattr__('conv1x3')
        self.__delattr__('conv1x1_extra')
        
        self.is_repped = True
        print("🚀 自动切换到推理模式 - [完全融合] 所有分支已合并为单一 5x5 卷积")

    def get_equivalent_kernel_bias(self):
        # 从一个确定存在的参数获取设备信息
        device = self.conv1x1[1].weight.device
        
        # 最终融合核的尺寸由最大的分支决定 (5x5)
        kernel_size = 5
        kernel = torch.zeros(self.out_channels, self.in_channels, kernel_size, kernel_size, device=device)
        bias = torch.zeros(self.out_channels, device=device)
        
        # 统一处理所有分支
        all_branches = {
            'dwconv5x5': self.dwconv5x5,
            'conv1x1': self.conv1x1,
            'conv3x1': self.conv3x1,
            'conv1x3': self.conv1x3,
            'conv1x1_extra': self.conv1x1_extra
        }
        
        for name, branch in all_branches.items():
            if name == 'dwconv5x5':
                k, b = self._fuse_dwconv_bn(branch)
            else:
                k, b = self._fuse_conv_bn(branch)
            
            kernel += self._pad_kernel_to_size(k, kernel_size)
            bias += b
        return kernel, bias

    # --- 融合辅助函数 ---
    def _fuse_dwconv_bn(self, dwconv_bn_seq):
        dw_block, bn = dwconv_bn_seq[0], dwconv_bn_seq[1]
        dw_conv, pw_conv = dw_block.depthwise_conv, dw_block.pointwise_conv
        pw_k_fused, pw_b_fused = self._fuse_conv_bn_manual(pw_conv, bn)
        dw_k = dw_conv.weight
        dw_k_reshaped = dw_k.transpose(0, 1)
        final_kernel = pw_k_fused * dw_k_reshaped
        return final_kernel, pw_b_fused

    def _fuse_conv_bn_manual(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_conv_bn(self, conv_bn_seq):
        return self._fuse_conv_bn_manual(conv_bn_seq[0], conv_bn_seq[1])

    def _pad_kernel_to_size(self, kernel, target_size):
        kernel_h, kernel_w = kernel.size(2), kernel.size(3)
        assert kernel_h <= target_size and kernel_w <= target_size
        pad_h = target_size - kernel_h
        pad_w = target_size - kernel_w
        padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        return F.pad(kernel, padding)

class RepFusionBlock(nn.Module):
    """
    可重参数化融合块 (RepFusionBlock) - 完整实现版
    
    该模块在训练时利用多样的并行分支增强模型性能，
    在推理时可等效融合成一个单一的卷积层以实现极致加速。
    它融合了 RepVGG, ACNet, 和 DBB 的核心思想。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.deploy = deploy

        self.activation = nn.ReLU(inplace=True)

        if self.deploy:
            # 部署(推理)模式下，只保留一个融合后的卷积层
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                        padding=self.padding, bias=True)
        else:
            # 训练模式下，定义所有并行分支
            self.fused_conv = None

            # 分支1: RepVGG核心 - KxK主分支
            self.branch_kxk = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                        padding=self.padding, bias=False)
            self.bn_kxk = nn.BatchNorm2d(out_channels)

            # 分支2: RepVGG核心 - 1x1通道融合分支
            self.branch_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 
                                        padding=0, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)

            # 分支3 & 4: ACNet核心 - 非对称卷积分支
            self.branch_asym_h = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), stride,
                                           padding=(self.padding, 0), bias=False)
            self.bn_asym_h = nn.BatchNorm2d(out_channels)
            self.branch_asym_v = nn.Conv2d(in_channels, out_channels, (1, kernel_size), stride,
                                           padding=(0, self.padding), bias=False)
            self.bn_asym_v = nn.BatchNorm2d(out_channels)

            # 分支5: DBB核心 - 序列卷积分支
            # 使用与输入和输出通道数相同的中间通道数
            self.branch_sequential = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                # 使用与输出通道相同的组数来实现深度可分离卷积
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, 
                          padding=self.padding, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # 分支6: RepVGG核心 - 恒等映射分支
            if in_channels == out_channels and stride == 1:
                self.branch_identity = nn.BatchNorm2d(in_channels)
            else:
                self.branch_identity = None

    def _fuse_bn_tensor(self, conv, bn):
        """[原子操作] 将一个BatchNorm层的参数融合进它前面的卷积层。"""
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros_like(running_mean)
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_weight = w * t
        fused_bias = beta + (b - running_mean) * gamma / std
        return fused_weight, fused_bias

    def switch_to_deploy(self):
        """ 将所有训练时分支融合到单一的推理卷积层中。"""
        if self.deploy:
            return
        print(f"Fusing RepFusionBlock (in_ch={self.in_channels}, out_ch={self.out_channels}, k={self.kernel_size})")

        fused_kernel = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        fused_bias = torch.zeros(self.out_channels)
        
        # 融合 KxK 分支
        k_kxk, b_kxk = self._fuse_bn_tensor(self.branch_kxk, self.bn_kxk)
        fused_kernel += k_kxk
        fused_bias += b_kxk

        # 融合 1x1 分支
        k_1x1, b_1x1 = self._fuse_bn_tensor(self.branch_1x1, self.bn_1x1)
        fused_kernel += F.pad(k_1x1, [self.padding, self.padding, self.padding, self.padding])
        fused_bias += b_1x1

        # 融合 ACNet 非对称分支
        k_asym_h, b_asym_h = self._fuse_bn_tensor(self.branch_asym_h, self.bn_asym_h)
        fused_kernel += k_asym_h
        fused_bias += b_asym_h

        k_asym_v, b_asym_v = self._fuse_bn_tensor(self.branch_asym_v, self.bn_asym_v)
        fused_kernel += k_asym_v
        fused_bias += b_asym_v
        
        # 融合 Identity 恒等分支
        if self.branch_identity is not None:
            id_kernel = torch.zeros_like(self.branch_kxk.weight)
            for i in range(self.in_channels):
                id_kernel[i, i, self.padding, self.padding] = 1.0
            
            gamma = self.branch_identity.weight
            beta = self.branch_identity.bias
            running_mean = self.branch_identity.running_mean
            running_var = self.branch_identity.running_var
            eps = self.branch_identity.eps
            std = (running_var + eps).sqrt()
            
            fused_id_kernel = id_kernel * (gamma / std).reshape(-1, 1, 1, 1)
            fused_id_bias = beta - (running_mean * gamma / std)
            
            fused_kernel += fused_id_kernel
            fused_bias += fused_id_bias

        # 融合 DBB 序列分支
        conv1, bn1, conv2, bn2 = self.branch_sequential
        k1_fused, b1_fused = self._fuse_bn_tensor(conv1, bn1)
        k2_fused, b2_fused = self._fuse_bn_tensor(conv2, bn2)
        
        # 核心：将两个融合后的卷积层再融合成一个
        # 等效于对卷积核进行卷积操作: K_final = conv(K1, K2)
        # k1_fused shape: (out, in, 1, 1)
        # k2_fused shape: (out, 1, K, K) (因为是depthwise)
        # 我们需要将k1_fused的输出通道(out_channels)作为k2_fused的输入通道
        # 这可以通过对k1_fused进行维度变换，并使用分组卷积巧妙实现
        final_seq_kernel = F.conv2d(
            input=k1_fused.permute(1, 0, 2, 3), # (in, out, 1, 1)
            weight=k2_fused, # (out, 1, K, K)
            stride=1,
            padding=self.padding,
            groups=self.out_channels
        ).permute(1, 0, 2, 3) # -> (out, in, K, K)

        # 计算融合后的偏置: B_final = conv(B1, K2) + B2
        final_seq_bias = F.conv2d(b1_fused.reshape(1, -1, 1, 1), k2_fused, groups=self.out_channels).reshape(-1) + b2_fused
        
        fused_kernel += final_seq_kernel
        fused_bias += final_seq_bias
        
        # 将最终融合的权重和偏置赋值给部署时的卷积层
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                    padding=self.padding, bias=True)
        self.fused_conv.weight.data = fused_kernel.to(self.fused_conv.weight.device)
        self.fused_conv.bias.data = fused_bias.to(self.fused_conv.bias.device)
        
        self._remove_training_branches()
        self.deploy = True

    def _remove_training_branches(self):
        """清理函数：删除所有训练时使用的分支以节省内存"""
        if hasattr(self, 'branch_kxk'): self.__delattr__('branch_kxk')
        if hasattr(self, 'bn_kxk'): self.__delattr__('bn_kxk')
        if hasattr(self, 'branch_1x1'): self.__delattr__('branch_1x1')
        if hasattr(self, 'bn_1x1'): self.__delattr__('bn_1x1')
        if hasattr(self, 'branch_asym_h'): self.__delattr__('branch_asym_h')
        if hasattr(self, 'bn_asym_h'): self.__delattr__('bn_asym_h')
        if hasattr(self, 'branch_asym_v'): self.__delattr__('branch_asym_v')
        if hasattr(self, 'bn_asym_v'): self.__delattr__('bn_asym_v')
        if hasattr(self, 'branch_sequential'): self.__delattr__('branch_sequential')
        if hasattr(self, 'branch_identity'): self.__delattr__('branch_identity')

    def forward(self, x):
        if self.deploy:
            return self.activation(self.fused_conv(x))
        else:
            out_kxk = self.bn_kxk(self.branch_kxk(x))
            out_1x1 = self.bn_1x1(self.branch_1x1(x))
            out_asym_h = self.bn_asym_h(self.branch_asym_h(x))
            out_asym_v = self.bn_asym_v(self.branch_asym_v(x))
            out_seq = self.branch_sequential(x)
            out_identity = self.branch_identity(x) if self.branch_identity is not None else 0
            
            return self.activation(out_kxk + out_1x1 + out_asym_h + out_asym_v + out_seq + out_identity)

class RepFusionBlockLite(nn.Module):
    """
    RepFusionBlock的轻量化版本，只保留核心的RepVGG三分支。
    此为完整实现，包含完整的部署切换逻辑。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        
        self.activation = nn.ReLU(inplace=True)

        if self.deploy:
            # 部署模式下，只定义一个融合后的卷积层
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                        padding=self.padding, bias=True)
        else:
            # 训练模式下，定义所有并行分支
            self.fused_conv = None

            self.branch_kxk = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                        padding=self.padding, bias=False)
            self.bn_kxk = nn.BatchNorm2d(out_channels)

            self.branch_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 
                                        padding=0, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)

            if in_channels == out_channels and stride == 1:
                self.branch_identity = nn.BatchNorm2d(in_channels)
            else:
                self.branch_identity = None

    def _fuse_bn_tensor(self, conv, bn):
        """[辅助函数] 将一个BatchNorm层的参数融合进它前面的卷积层。"""
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros_like(running_mean)
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_weight = w * t
        fused_bias = beta + (b - running_mean) * gamma / std
        return fused_weight, fused_bias

    def switch_to_deploy(self):
        """[核心函数] 将所有训练时分支融合到单一的推理卷积层中。"""
        if self.deploy:
            return
        
        print(f"Fusing RepFusionBlockLite (in_ch={self.in_channels}, out_ch={self.out_channels}, k={self.kernel_size})")

        # 从模块的参数中同时获取正确的设备和数据类型
        example_param = self.branch_kxk.weight
        target_device = example_param.device
        target_dtype = example_param.dtype
        
        # 在创建累加器张量时，明确指定 device 和 dtype
        fused_kernel = torch.zeros(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 
            device=target_device, dtype=target_dtype
        )
        fused_bias = torch.zeros(self.out_channels, device=target_device, dtype=target_dtype)

        # 1. 融合 KxK 分支
        k_kxk, b_kxk = self._fuse_bn_tensor(self.branch_kxk, self.bn_kxk)
        fused_kernel += k_kxk
        fused_bias += b_kxk

        # 2. 融合 1x1 分支
        k_1x1, b_1x1 = self._fuse_bn_tensor(self.branch_1x1, self.bn_1x1)
        fused_kernel += F.pad(k_1x1, [self.padding, self.padding, self.padding, self.padding])
        fused_bias += b_1x1

        # 3. 融合 Identity 恒等分支
        if self.branch_identity is not None:
            id_kernel = torch.zeros_like(self.branch_kxk.weight, dtype=target_dtype) # 同样指定dtype
            for i in range(self.in_channels):
                id_kernel[i, i, self.padding, self.padding] = 1.0
            
            gamma, beta, mean, var = self.branch_identity.weight, self.branch_identity.bias, self.branch_identity.running_mean, self.branch_identity.running_var
            eps = self.branch_identity.eps
            std = (var + eps).sqrt()
            
            fused_id_kernel = id_kernel * (gamma / std).reshape(-1, 1, 1, 1)
            fused_id_bias = beta - (mean * gamma / std)
            
            fused_kernel += fused_id_kernel
            fused_bias += fused_id_bias
        
        # 4. 创建并赋值给最终的推理卷积层
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            padding=self.padding, bias=True
        )
        self.fused_conv.to(target_device, dtype=target_dtype) # 确保新层也符合设备和类型
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        
        # 5. 清理训练时的分支，切换为部署模式
        self._remove_training_branches()
        self.deploy = True

    def _remove_training_branches(self):
        """[辅助函数] 清理所有训练时使用的分支以节省内存"""
        if hasattr(self, 'branch_kxk'): self.__delattr__('branch_kxk')
        if hasattr(self, 'bn_kxk'): self.__delattr__('bn_kxk')
        if hasattr(self, 'branch_1x1'): self.__delattr__('branch_1x1')
        if hasattr(self, 'bn_1x1'): self.__delattr__('bn_1x1')
        if hasattr(self, 'branch_identity'): self.__delattr__('branch_identity')

    def forward(self, x):
        if self.deploy:
            return self.activation(self.fused_conv(x))
        else:
            out_kxk = self.bn_kxk(self.branch_kxk(x))
            out_1x1 = self.bn_1x1(self.branch_1x1(x))
            out_identity = self.branch_identity(x) if self.branch_identity is not None else 0
            return self.activation(out_kxk + out_1x1 + out_identity)

if __name__ == '__main__':
    from attention import *
    # x = torch.randn(16, 3, 256, 256).to('cuda')
    # # model = SCS_Mamba(dim=3, d_state=64, d_conv=4).to('cuda')
    # model = RepFusionBlock(in_channels=3, out_channels=3, kernel_size=3).to('cuda')
    # out = model(x)
    # print(out.shape, "\n",
    #       model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSFN_Rep(in_channels=64, out_channels=64).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    
    print("--- 步骤 1: 模拟训练并稳定BN统计量 ---")
    model.train()
    # 在真实训练中，你会有一个循环。这里我们只做几次前向传播来模拟。
    for _ in range(3):
        model(torch.randn(2, 64, 32, 32).to(device))
    print("BN 运行统计量已更新。")

    print("\n--- 步骤 2: 计算训练模式下的输出 (基准) ---")
    # 确保模型仍在训练模式
    model.train()
    with torch.no_grad(): # 在no_grad下计算，避免BN统计量再次变化
        out_train = model(x)
    print(f"训练输出形状: {out_train.shape}")

    print("\n--- 步骤 3: 切换到推理模式并计算输出 ---")
    # 调用eval()将触发重参数化
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"推理输出形状: {out_eval.shape}")
    
    print("\n--- 步骤 4: 验证等价性 ---")
    difference = torch.dist(out_train, out_eval)
    print(f"输出差异 (应非常小): {difference.item()}")
    
    # 检查差异是否在可接受的浮点误差范围内
    if torch.allclose(out_train, out_eval, atol=1e-5):
        print("✅ 验证成功！训练和推理模式输出等价！")
    else:
        print("❌ 验证失败！输出不等价，请检查融合逻辑。")

elif os.path.dirname(os.path.abspath(__file__)) == '/mnt/e/VScode/WS-Hub/WS-UNet/UNet/models/utils':
    from .attention import *

else:
    from src.models.utils import *

        
     

        