import torch
from model.resnet import resnet50
import torch.nn as nn
from model.utils.modules import *

class Res_UNet(nn.Module):
    def __init__(self, 
                 inchannels,
                 num_classes,
                 p,
                 basechannel=32):
        super().__init__()
        self.down = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder = resnet50(pretrained=True, inchannels=3, num_classes=basechannel*8, return_more = True, deep_base=False)
        # 解码器
        self.decoder1 = DoubleConv(basechannel * 16, basechannel * 4)
        self.decoder2 = DoubleConv(basechannel * 8,  basechannel * 2)
        self.decoder3 = DoubleConv(basechannel * 4,  basechannel)
        self.decoder4 = DoubleConv(basechannel * 2,  basechannel)
        # 输出层
        self.out_conv = nn.Conv2d(basechannel, num_classes)

    def forward(self, x):
        c5, x1, x2, x3, x4 = self.encoder(x) # 256 256 128 64 64
        
        d4 = self.up(c5)
        d4 = torch.cat([d4,x4], dim=1)       # 256 + 
        d4 = self.decoder1(d4)               # 

        d3 = self.up(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.decoder2(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.decoder3(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.decoder4(d1)

        logits = self.out_conv(d1)
        return logits