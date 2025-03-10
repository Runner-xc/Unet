import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter
from model.utils.attention import *
from model.utils.modules import * 

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class MSAF_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(MSAF_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels
        # 编码器
        self.inconv = DoubleConv(in_channels, base_channels)
        self.acpn0 = ACPN(base_channels)
        self.msaf1 = EMAF(base_channels) 

        self.down1 = ResD_Down(base_channels, base_channels*2)
        self.acpn1 = ACPN(base_channels*2)       
        self.msaf2 = EMAF(base_channels*2)

        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.acpn2 = ACPN(base_channels*4) 
        self.msaf3 = EMAF(base_channels*4) 

        self.down3 = ResD_Down(base_channels*4, base_channels*8)
        self.acpn3 = ACPN(base_channels*8)
        self.msaf4 = EMAF(base_channels*8)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        # self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        
        # 解码器
        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.acpn5 = ACPN(base_channels*4)

        self.up2 = ResD_Up(base_channels * 4, base_channels * 2 , bilinear=bilinear)
        self.acpn6 = ACPN(base_channels*2)    
     
        self.up3 = ResD_Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.acpn7 = ACPN(base_channels)

        self.up4 = Up(base_channels, base_channels, bilinear=bilinear)
        self.acpn8 = ACPN(base_channels)
        # 输出层
        self.out_conv = OutConv(base_channels, n_classes)

    def forward(self, x):
        e1 = self.inconv(x)
        x1 = self.acpn0(e1)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]

        e2 = self.down1(x1)                              # [b, 64, 128, 128]
        a2 = self.acpn1(e2)
        m2 = self.msaf2(e2)
        x2 = self.encoder_dropout(a2)
        
        e3 = self.down2(x2)                              # [b, 128, 64, 64]
        a3 = self.acpn2(e3)
        m3 = self.msaf3(e3)
        x3 = self.encoder_dropout(a3)

        e4 = self.down3(x3)                              # [b, 256, 32, 32]
        a4 = self.acpn3(e4)
        m4 = self.msaf4(e4)
        x4 = self.encoder_dropout(a4)
        
        x5 = self.pool(x4)
        # x = self.dense_aspp(x)                         # [b, 512, 16, 16]   
        c5 = self.center_conv(x5)                        # [b, 512, 16, 16]
        c5 = self.bottleneck_dropout(c5)

        d4 = self.up1(c5, m4)                            # [b, 256, 32, 32]
        d4 = self.acpn5(d4)
        x = self.decoder_dropout(d4)   

        d3 = self.up2(x, m3)                             # [b, 128, 64, 64]
        d3 = self.acpn6(d3)
        x = self.decoder_dropout(d3)

        d2 = self.up3(x, m2)                             # [b, 64, 128, 128]
        d2 = self.acpn7(d2)
        x = self.decoder_dropout(d2)

        d1 = self.up4(x, m1)                             # [1, 64, 256, 256]
        d1 = self.acpn8(d1)
        x = self.decoder_dropout(d1)
        logits = self.out_conv(x)                        # [1, c, 256, 256]
        
        return logits
    
class MSAF_UNetv2(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(MSAF_UNetv2, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels
        # 编码器
        self.inconv = DoubleConv(in_channels, base_channels)
        self.acpn0 = ACPNv2(base_channels)
        self.msaf1 = EMAF(base_channels) 

        self.down1 = ResD_Down(base_channels, base_channels*2)
        self.acpn1 = ACPNv2(base_channels*2)       
        self.msaf2 = EMAF(base_channels*2)

        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.acpn2 = ACPNv2(base_channels*4) 
        self.msaf3 = EMAF(base_channels*4) 

        self.down3 = ResD_Down(base_channels*4, base_channels*8)
        self.acpn3 = ACPNv2(base_channels*8)
        self.msaf4 = EMAF(base_channels*8)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        # self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        
        # 解码器
        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.acpn5 = ACPNv2(base_channels*4)

        self.up2 = ResD_Up(base_channels * 4, base_channels * 2 , bilinear=bilinear)
        self.acpn6 = ACPNv2(base_channels*2)    
     
        self.up3 = ResD_Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.acpn7 = ACPNv2(base_channels)

        self.up4 = Up(base_channels, base_channels, bilinear=bilinear)
        self.acpn8 = ACPNv2(base_channels)
        # 输出层
        self.out_conv = OutConv(base_channels, n_classes)

    def forward(self, x):
        e1 = self.inconv(x)
        x1 = self.acpn0(e1)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]

        e2 = self.down1(x1)                              # [b, 64, 128, 128]
        e2 = self.encoder_dropout(e2)
        a2 = self.acpn1(e2)
        m2 = self.msaf2(e2)
        x2 = self.encoder_dropout(a2)
        
        e3 = self.down2(x2)                              # [b, 128, 64, 64]
        e3 = self.encoder_dropout(e3)
        a3 = self.acpn2(e3)
        m3 = self.msaf3(e3)
        x3 = self.encoder_dropout(a3)

        e4 = self.down3(x3)                              # [b, 256, 32, 32]
        e4 = self.encoder_dropout(e4)
        a4 = self.acpn3(e4)
        m4 = self.msaf4(e4)
        x4 = self.encoder_dropout(a4)
        
        x5 = self.pool(x4)
        # x = self.dense_aspp(x)                         # [b, 512, 16, 16]   
        c5 = self.center_conv(x5)                        # [b, 512, 16, 16]
        c5 = self.bottleneck_dropout(c5)

        d4 = self.up1(c5, m4)                            # [b, 256, 32, 32]
        d4 = self.encoder_dropout(d4)
        d4 = self.acpn5(d4)
        x = self.decoder_dropout(d4)   

        d3 = self.up2(x, m3)                             # [b, 128, 64, 64]
        d3 = self.encoder_dropout(d3)
        d3 = self.acpn6(d3)
        x = self.decoder_dropout(d3)

        d2 = self.up3(x, m2)                             # [b, 64, 128, 128]
        d2 = self.encoder_dropout(d2)
        d2 = self.acpn7(d2)
        x = self.decoder_dropout(d2)

        d1 = self.up4(x, m1)                             # [1, 64, 256, 256]
        d1 = self.encoder_dropout(d1)
        d1 = self.acpn8(d1)
        x = self.decoder_dropout(d1)
        logits = self.out_conv(x)                        # [1, c, 256, 256]
        
        return logits