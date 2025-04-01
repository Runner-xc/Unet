import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from model.utils.modules import DoubleConv

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class AIConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], directions=['x','y']):
        super(AIConv2d, self).__init__()
        self.directions = directions
        self.kernel_sizes = kernel_sizes
        self.relu = nn.ReLU(inplace=True)
        self.bcnorm = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 每个方向的多核卷积层
        self.direction_convs = nn.ModuleDict()
        for direction in directions:
            convs = nn.ModuleList()
            for ksize in kernel_sizes:
                # 沿不同方向的一维卷积 
                if direction == 'x':
                    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(ksize,1), padding=(ksize//2,0))
                elif direction == 'y':
                    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,ksize), padding=(0,ksize//2))
                convs.append(conv)
            self.direction_convs[direction] = convs
            
        # 调制因子生成器 (轻量级映射)
        self.fc_modulation = nn.Sequential(
            nn.Conv2d(in_channels, len(directions)*len(kernel_sizes), kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化生成空间无关的权重
            nn.Flatten()
        )
        
    def forward(self, x):
        # Step 1: 各方向多核卷积并行计算
        direction_outputs = {}
        for direction in self.directions:
            conv_outputs = []
            for conv in self.direction_convs[direction]:
                conv_out = conv(x)
                conv_outputs.append(conv_out)
            direction_outputs[direction] = torch.stack(conv_outputs, dim=1)  # [B, K, C, H, W]
        
        # Step 2: 生成调制因子 (动态权重)
        B, _, H, W = x.shape
        modulation = self.fc_modulation(x)  # [B, D*K]
        modulation = modulation.view(B, len(self.directions), len(self.kernel_sizes))
        modulation = F.softmax(modulation, dim=2)  # 方向间独立Softmax
        
        # Step 3: 加权融合各方向结果
        combined = []
        for i, direction in enumerate(self.directions):
            weights = modulation[:, i, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weighted = (direction_outputs[direction] * weights).sum(dim=1)  # 加权求和
            weighted = self.relu(weighted)
            combined.append(weighted)
        
        # Step 4: 残差连接
        x = self.conv1x1(x)
        output = sum(combined) + x  # 残差连接
        output = self.bcnorm(output)
        output = self.relu(output)
        return output
    
class AIConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], directions=['x','y','z']):
        super(AIConv3d, self).__init__()
        self.directions = directions
        self.kernel_sizes = kernel_sizes
        self.relu = nn.ReLU(inplace=True)
        self.bcnorm = nn.BatchNorm3d(out_channels)
        self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        # 每个方向的多核卷积层
        self.direction_convs = nn.ModuleDict()
        for direction in directions:
            convs = nn.ModuleList()
            for ksize in kernel_sizes:
                # 沿不同方向的一维卷积 (e.g., x方向为3D中的depth-wise)
                if direction == 'x':
                    conv = nn.Conv3d(in_channels, out_channels, kernel_size=(ksize,1,1), padding=(ksize//2,0,0))
                elif direction == 'y':
                    conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,ksize,1), padding=(0,ksize//2,0))
                elif direction == 'z':
                    conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,ksize), padding=(0,0,ksize//2))
                convs.append(conv)
            self.direction_convs[direction] = convs
            
        # 调制因子生成器 (轻量级映射)
        self.fc_modulation = nn.Sequential(
            nn.Conv3d(in_channels, len(directions)*len(kernel_sizes), kernel_size=1),
            nn.AdaptiveAvgPool3d(1),  # 全局平均池化生成空间无关的权重
            nn.Flatten()
        )
        
    def forward(self, x):
        # Step 1: 各方向多核卷积并行计算
        direction_outputs = {}
        for direction in self.directions:
            conv_outputs = []
            for conv in self.direction_convs[direction]:
                conv_out = conv(x)
                conv_outputs.append(conv_out)
            direction_outputs[direction] = torch.stack(conv_outputs, dim=1)  # [B, K, C, D, H, W]
        
        # Step 2: 生成调制因子 (动态权重)
        B, _, D, H, W = x.shape
        modulation = self.fc_modulation(x)  # [B, D*K]
        modulation = modulation.view(B, len(self.directions), len(self.kernel_sizes))
        modulation = F.softmax(modulation, dim=2)  # 方向间独立Softmax
        
        # Step 3: 加权融合各方向结果
        combined = []
        for i, direction in enumerate(self.directions):
            weights = modulation[:, i, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weighted = (direction_outputs[direction] * weights).sum(dim=1)  # 加权求和
            weighted = self.relu(weighted)
            combined.append(weighted)
        
        # Step 4: 残差连接
        x = self.conv1x1x1(x)
        output = sum(combined) + x  # 残差连接
        output = self.bcnorm(output)
        output = self.relu(output)
        return output
    
class AICUNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(AICUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        self.encoder1 = AIConv2d(in_channels, base_channels) 
        self.encoder2 = AIConv2d(base_channels, base_channels*2)
        self.encoder3 = AIConv2d(base_channels*2, base_channels*4)
        self.encoder4 = AIConv2d(base_channels*4, base_channels*8)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        
        self.decoder1 = AIConv2d(base_channels * 16, base_channels * 4)
        self.decoder2 = AIConv2d(base_channels * 8, base_channels * 2)
        self.decoder3 = AIConv2d(base_channels * 4, base_channels)
        self.decoder4 = AIConv2d(base_channels * 2, base_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.encoder1(x)             # [1, 32, 320, 320]

        x2 = self.down(x1)
        x2 = self.encoder2(x2)             # [1, 64, 160, 160]
        x2 = self.encoder_dropout(x2)

        x3 = self.down(x2)        
        x3 = self.encoder3(x3)             # [1, 128, 80, 80]
        x3 = self.encoder_dropout(x3)

        x4 = self.down(x3)
        x4 = self.encoder4(x4)             # [1, 256, 40, 40]
        x4 = self.encoder_dropout(x4)

        x5 = self.down(x4)
        x5 = self.encoder_dropout(x5)
                   
        x = self.center_conv(x5)        # [1, 256, 20, 20]
        x = self.bottleneck_dropout(x)
        
        x = self.up(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder1(x)             # [1, 512, 40, 40]
        x = self.decoder_dropout(x)

        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder2(x)             # [1, 128, 80, 80]
        x = self.decoder_dropout(x)

        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder3(x)             # [1, 64, 160, 160]
        x = self.decoder_dropout(x)

        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder4(x)             # [1, 32, 320, 320]
        x = self.decoder_dropout(x)

        logits = self.out_conv(x)       # [1, c, 320, 320]       
        return logits 
    
if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256, 256)
    # model1 = AIConv2d(3, 3)
    # model2 = AIConv3d(3, 3)
    # out1 = model1(x)
    # out2 = model2(y)
    # print(x.shape)
    # print(y.shape)
    aicunet = AICUNet(in_channels=3, n_classes=4, p=0)
    output = aicunet(x)
    summary(aicunet)
    print(output.shape)