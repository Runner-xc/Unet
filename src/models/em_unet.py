import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter 
from mamba_ssm import Mamba


class RepConvBlock(nn.Module):
    """
    结构重参数化卷积块 (受RepVGG启发).
    训练时采用多分支结构，推理时融合成单一的3x3卷积.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 推理时使用的融合后卷积
        self.rep_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # 训练时的三个分支
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding - kernel_size // 2)
        if in_channels == out_channels:
            self.identity_branch = nn.BatchNorm2d(in_channels)
        else:
            self.identity_branch = None

    def forward(self, x):
        if self.training:
            # 训练时，计算所有分支
            out_3x3 = self.conv_3x3(x)
            out_1x1 = self.conv_1x1(x)
            out_identity = self.identity_branch(x) if self.identity_branch is not None else 0
            
            # 融合BN层到卷积权重 (这是get_equivalent_kernel_bias做的事情，这里为概念演示)
            # 在实际使用中，我们需要一个专门的函数来处理融合
            # 为简单起见，这里直接相加
            out = out_3x3 + out_1x1 + out_identity
            return self.activation(self.bn(out))
        else:
            # 推理时，只使用融合后的卷积
            return self.activation(self.rep_conv(x))
        
class CrossScanMambaBlock(nn.Module):
    """
    交叉扫描Mamba瓶颈层。
    从四个方向扫描特征图以聚合全局上下文。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, n_layers=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim

        # 准备一个残差连接
        residual = x

        # 1. 将输入展平并归一化
        x_flat = x.reshape(B, C, H * W).transpose(1, 2)
        x_norm = self.norm(x_flat)

        # --- 多方向扫描 ---
        outputs = []
        for i in range(self.n_layers):
            mamba_layer = self.mamba_layers[i]
            
            # 方向1: 正向扫描
            if i % 4 == 0:
                scan_input = x_norm
            # 方向2: 反向扫描 (通过反转序列实现)
            elif i % 4 == 1:
                scan_input = torch.flip(x_norm, dims=[1])
            # 方向3: 垂直正向扫描 (通过转置H,W实现)
            elif i % 4 == 2:
                x_transposed = x_norm.reshape(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H*W, C)
                scan_input = x_transposed
            # 方向4: 垂直反向扫描
            else:
                x_transposed_flipped = torch.flip(x_norm.reshape(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H*W, C), dims=[1])
                scan_input = x_transposed_flipped

            # 应用Mamba
            scan_output = mamba_layer(scan_input)

            # --- 恢复原始顺序和形状 ---
            if i % 4 == 0:
                output_i = scan_output
            elif i % 4 == 1:
                output_i = torch.flip(scan_output, dims=[1])
            elif i % 4 == 2:
                output_i = scan_output.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H*W, C)
            else:
                output_i = torch.flip(scan_output.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H*W, C), dims=[1])
            
            outputs.append(output_i)

        # 聚合所有方向的输出
        x_agg = sum(outputs) # 简单地相加是一种有效的聚合方式
        
        # 恢复形状并加上残差
        x_out = x_agg.transpose(1, 2).reshape(B, C, H, W)
        x_out = x_out + residual
        
        return x_out

class EM_UNet(nn.Module):
    def __init__(self, 
                 in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 ):
        super(EM_UNet, self).__init__()
        
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = RepConvBlock(in_channels, base_channels)
        self.encoder2 = RepConvBlock(base_channels, base_channels*2)
        self.encoder3 = RepConvBlock(base_channels*2, base_channels*4)
        self.encoder4 = RepConvBlock(base_channels*4, base_channels*8)
        # encoder_dropout
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0)
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)
        # bottleneck
        self.center_conv = CrossScanMambaBlock(base_channels*8)
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0)
        # att
        self.att_gate1 = SCS_Mamba(base_channels,   d_state=16, d_conv=4, expand=2)
        self.att_gate2 = SCS_Mamba(base_channels*2, d_state=16, d_conv=4, expand=2)
        self.att_gate3 = SCS_Mamba(base_channels*4, d_state=16, d_conv=4, expand=2)
        self.att_gate4 = SCS_Mamba(base_channels*8, d_state=16, d_conv=4, expand=2) 

        # 解码器
        self.decoder1 = RepConvBlock(base_channels * 16 , base_channels * 4) 
        self.decoder2 = RepConvBlock(base_channels * 8 ,  base_channels * 2)
        self.decoder3 = RepConvBlock(base_channels * 4 ,  base_channels)
        self.decoder4 = RepConvBlock(base_channels * 2,   base_channels)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)               # [1, 32, 320, 320]
        x2 = self.down(x1)
        x2 = self.encoder_dropout1(x2)

        x2 = self.encoder2(x2)              # [1, 64, 160, 160]
        x3 = self.down(x2)
        x3 = self.encoder_dropout2(x3)

        x3 = self.encoder3(x3)              # [1, 128, 80, 80]
        x4 = self.down(x3)
        x4 = self.encoder_dropout3(x4)

        x4 = self.encoder4(x4)              # [1, 256, 40, 40]
        x5 = self.down(x4)
        x5 = self.encoder_dropout4(x5)
        
        x = self.center_conv(x5)            # [1, 256, 20, 20]
        x = self.bottleneck_dropout(x)
        
        x = self.up(x)                      # [1, 256, 40, 40]
        x4 = self.att_gate4(x4)
        x = torch.cat([x, x4], dim=1)       # [1, 256+256, 40, 40]            
        x = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up(x)                      # [1, 128, 80, 80]
        x3 = self.att_gate3(x3)
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        x = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up(x)                      # [1, 64, 160, 160]
        x2 = self.att_gate2(x2)
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        x = self.decoder3(x)                # [1, 32, 160, 160]

        x = self.up(x)                      # [1, 32, 160, 160]
        x1 = self.att_gate1(x1)
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        
        logits = self.out_conv(x)           # [1, c, 320, 320]       
        return logits
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss

class EM_UNetv2(EM_UNet):
    def __init__(self, 
                 in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 ):
        super(EM_UNetv2, self).__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            p=p,
            base_channels=base_channels)
        # 编码器
        self.encoder1 = RepFusionBlockLite(in_channels,     base_channels)
        self.encoder2 = RepFusionBlockLite(base_channels,   base_channels*2)
        self.encoder3 = RepFusionBlockLite(base_channels*2, base_channels*4)
        self.encoder4 = RepFusionBlockLite(base_channels*4, base_channels*8)
        # 解码器
        self.decoder1 = RepFusionBlockLite(base_channels * 16 , base_channels * 4) 
        self.decoder2 = RepFusionBlockLite(base_channels * 8 ,  base_channels * 2)
        self.decoder3 = RepFusionBlockLite(base_channels * 4 ,  base_channels)
        self.decoder4 = RepFusionBlockLite(base_channels * 2,   base_channels)
    
    def forward(self, x):
        return super().forward(x)

class EM_UNetv3(EM_UNet):
    def __init__(self, 
                 in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 ):
        super(EM_UNetv3, self).__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            p=p,
            base_channels=base_channels)
        # 编码器
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels*2)
        self.encoder3 = DoubleConv(base_channels*2, base_channels*4)
        self.encoder4 = DoubleConv(base_channels*4, base_channels*8)
        # bottleneck
        self.center_conv = SCS_Mamba(base_channels*8, d_state=16, d_conv=4, expand=2)
        # att
        # self.att1 = SCS_Mamba(base_channels,   d_state=16, d_conv=4, expand=2)
        # self.att2 = SCS_Mamba(base_channels*2, d_state=16, d_conv=4, expand=2)
        # self.att3 = SCS_Mamba(base_channels*4, d_state=16, d_conv=4, expand=2)
        # self.att4 = SCS_Mamba(base_channels*8, d_state=16, d_conv=4, expand=2) 
        
        self.att_gate1 = Att_gate(base_channels,   base_channels,   base_channels//2)
        self.att_gate2 = Att_gate(base_channels*2, base_channels*2, base_channels)
        self.att_gate3 = Att_gate(base_channels*4, base_channels*4, base_channels*2)
        self.att_gate4 = Att_gate(base_channels*8, base_channels*8, base_channels*4)
        # 解码器
        self.decoder1 = DoubleConv(base_channels * 16 , base_channels * 4) 
        self.decoder2 = DoubleConv(base_channels * 8 ,  base_channels * 2)
        self.decoder3 = DoubleConv(base_channels * 4 ,  base_channels)
        self.decoder4 = DoubleConv(base_channels * 2,   base_channels)
    def forward(self, x):
        x1 = self.encoder1(x)               # [1, 32, 320, 320]
        x2 = self.down(x1)
        x2 = self.encoder_dropout1(x2)

        x2 = self.encoder2(x2)              # [1, 64, 160, 160]
        x3 = self.down(x2)
        x3 = self.encoder_dropout2(x3)

        x3 = self.encoder3(x3)              # [1, 128, 80, 80]
        x4 = self.down(x3)
        x4 = self.encoder_dropout3(x4)

        x4 = self.encoder4(x4)              # [1, 256, 40, 40]
        x5 = self.down(x4)
        x5 = self.encoder_dropout4(x5)
        
        x = self.center_conv(x5)            # [1, 256, 20, 20]
        x = self.bottleneck_dropout(x)
        
        x = self.up(x)                      # [1, 256, 40, 40]
        # x4 = self.att4(x4)
        x4 = self.att_gate4(x, x4)
        x = torch.cat([x, x4], dim=1)       # [1, 256+256, 40, 40]            
        x = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up(x)                      # [1, 128, 80, 80]
        # x3 = self.att3(x3)
        x3 = self.att_gate3(x, x3)
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        x = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up(x)                      # [1, 64, 160, 160]
        # x2 = self.att2(x2)
        x2 = self.att_gate2(x, x2)
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        x = self.decoder3(x)                # [1, 32, 160, 160]

        x = self.up(x)                      # [1, 32, 160, 160]
        # x1 = self.att1(x1)
        x1 = self.att_gate1(x, x1)
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        
        logits = self.out_conv(x)           # [1, c, 320, 320]       
        return logits

    
if __name__ == '__main__':
    from utils import *      
    model = EM_UNetv3(in_channels=3, num_classes=4, p=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, (8, 3, 256, 256))
    calculate_computation(model, input_size=(3, 256, 256), device=device)
    # ========================================
    # Input size: (3, 256, 256)
    # FLOPs: 28.87 GFLOPs
    # MACs: 14.44 GMACs
    # Params: 9.04 M
    # ========================================

else:
    from src.models.utils import *