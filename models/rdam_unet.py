import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter 

      
class RDAM_UNet(nn.Module):
    def __init__(self, 
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(RDAM_UNet, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 编码器
        self.encoder1 = DoubleConv(in_channels,     base_channels  )
        self.encoder2 = ResDConv(base_channels,     base_channels*2)
        self.encoder3 = ResDConv(base_channels*2,   base_channels*4)
        self.encoder4 = ResDConv(base_channels*4,   base_channels*8)
        # 多尺度融合
        self.acpn0 = AMSFN(base_channels)
        self.acpn1 = AMSFN(base_channels*2) 
        self.acpn2 = AMSFN(base_channels*4) 
        self.acpn3 = AMSFN(base_channels*8)
        # attention
        self.att1 = MDAM(base_channels) 
        self.att2 = MDAM(base_channels*2) 
        self.att3 = MDAM(base_channels*4)
        self.att4 = MDAM(base_channels*8)
        # encoder_dropout
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0) 
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)
                                    # 编码器更高dropout
        # bottleneck
        ## self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0)

        # 解码器
        self.decoder1 = ResDConv(base_channels * 16 ,   base_channels * 4)
        self.decoder2 = ResDConv(base_channels * 8,     base_channels * 2)
        self.decoder3 = ResDConv(base_channels * 4,     base_channels)
        self.decoder4 = DoubleConv(base_channels * 2,   base_channels)
        # 多尺度融合
        self.acpn5 = AMSFN(base_channels*4)
        self.acpn6 = AMSFN(base_channels*2) 
        self.acpn7 = AMSFN(base_channels) 
        self.acpn8 = AMSFN(base_channels)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)

        # 输出层
        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        m1 = self.att1(x1)
        x1 = self.acpn0(x1)
        x1 = self.encoder_dropout1(x1)

        x2 = self.down(x1)    
        x2 = self.encoder2(x2) 
        m2 = self.att2(x2)                              # [b, 64, 128, 128]
        x2 = self.acpn1(x2)
        x2 = self.encoder_dropout2(x2)

        x3 = self.down(x2)
        x3 = self.encoder3(x3)                              # [b, 128, 64, 64]
        m3 = self.att3(x3)
        x3 = self.acpn2(x3)
        x3 = self.encoder_dropout3(x3)

        x4 = self.down(x3)
        x4 = self.encoder4(x4)                              # [b, 256, 32, 32]
        m4 = self.att4(x4)
        x4 = self.acpn3(x4)
        x4 = self.encoder_dropout4(x4)

        x5 = self.down(x4)
        x5 = self.center_conv(x5)                           # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x5)

        d4 = self.up(x)
        d4 = torch.cat([d4, m4], dim=1)                     # [b, 256, 32, 32]
        d4 = self.decoder1(d4)
        d4 = self.acpn5(d4)
        x = self.decoder_dropout1(d4)   

        d3 = self.up(x)
        d3 = torch.cat([d3, m3], dim=1)                      # [b, 128, 64, 64]
        d3 = self.decoder2(d3)
        d3 = self.acpn6(d3)
        x = self.decoder_dropout2(d3)

        d2 = self.up(x)
        d2 = torch.cat([d2, m2], dim=1)                      # [b, 64, 128, 128]
        d2 = self.decoder3(d2)
        x = self.acpn7(d2)

        d1 = self.up(x)
        d1 = torch.cat([d1, m1], dim=1)                     # [b, 64, 256, 256]
        d1 = self.decoder4(d1)
        x = self.acpn8(d1)
        logits = self.out_conv(x)                           # [1, c, 256, 256]
        
        return logits
    
class DWRDAM_UNet(nn.Module):
    def __init__(self, 
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DWRDAM_UNet, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # attention
        self.att1 = MDAM(base_channels) 
        self.att2 = MDAM(base_channels*2) 
        self.att3 = MDAM(base_channels*4)
        self.att4 = MDAM(base_channels*8)

        # 编码器
        self.encoder1 = AMSFN(in_channels,     base_channels)
        self.encoder2 = AMSFN(base_channels,   base_channels*2)
        self.encoder3 = AMSFN(base_channels*2, base_channels*4) 
        self.encoder4 = AMSFN(base_channels*4, base_channels*8)  

        # self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
        self.center_conv = DWDoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)

        # 解码器
        self.decoder1 = DWDoubleConv(base_channels * 16 ,  base_channels * 4)
        self.decoder2 = DWDoubleConv(base_channels * 8,    base_channels * 2)
        self.decoder3 = DWDoubleConv(base_channels * 4,    base_channels,   )
        self.decoder4 = DWDoubleConv(base_channels * 2,    base_channels,   )
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

        # encoder_dropout
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0) 
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0)

    def forward(self, x):
        x1 = self.encoder1(x)
        m1 = self.att1(x1)
        x1 = self.encoder_dropout1(x1)

        x2 = self.down(x1)    
        x2 = self.encoder2(x2) 
        m2 = self.att2(x2)                              # [b, 64, 128, 128]
        x2 = self.encoder_dropout2(x2)

        x3 = self.down(x2)
        x3 = self.encoder3(x3)                              # [b, 128, 64, 64]
        m3 = self.att3(x3)
        x3 = self.encoder_dropout3(x3)

        x4 = self.down(x3)
        x4 = self.encoder4(x4)                              # [b, 256, 32, 32]
        m4 = self.att4(x4)
        x4 = self.encoder_dropout4(x4)

        x5 = self.down(x4)
        x5 = self.center_conv(x5)                           # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x5)

        d4 = self.up(x)
        d4 = torch.cat([d4, m4], dim=1)                     # [b, 256, 32, 32]
        x = self.decoder1(d4)
        x = self.decoder_dropout1(x)   

        d3 = self.up(x)
        d3 = torch.cat([d3, m3], dim=1)                      # [b, 128, 64, 64]
        x = self.decoder2(d3)
        x = self.decoder_dropout2(x)

        d2 = self.up(x)
        d2 = torch.cat([d2, m2], dim=1)                      # [b, 64, 128, 128]
        x = self.decoder3(d2)

        d1 = self.up(x)
        d1 = torch.cat([d1, m1], dim=1)                     # [b, 64, 256, 256]
        x = self.decoder4(d1)
        logits = self.out_conv(x)                           # [1, c, 256, 256]
        return logits

class DWRDAM_UNetV2(DWRDAM_UNet):
    def __init__(self, 
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DWRDAM_UNetV2, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            p=p,
            base_channels=base_channels
        )
        # 编码器
        self.encoder1 = AMSFNV2(in_channels,        base_channels)
        self.encoder2 = AMSFNV2(base_channels,      base_channels*2)
        self.encoder3 = AMSFNV2(base_channels*2,    base_channels*4) 
        self.encoder4 = AMSFNV2(base_channels*4,    base_channels*8) 

    def forward(self, x):
        return super(DWRDAM_UNetV2, self).forward(x)
    
class DWRDAM_UNetV3(DWRDAM_UNet):
    def __init__(self, 
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DWRDAM_UNetV3, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            p=p,
            base_channels=base_channels
        )
        # 编码器
        self.encoder1 = AMSFNV3(in_channels,        base_channels)
        self.encoder2 = AMSFNV3(base_channels,      base_channels*2)
        self.encoder3 = AMSFNV3(base_channels*2,    base_channels*4) 
        self.encoder4 = AMSFNV3(base_channels*4,    base_channels*8) 

    def forward(self, x):
        return super(DWRDAM_UNetV3, self).forward(x)
      
class DWRDAM_UNetV4(DWRDAM_UNetV2):
    def __init__(self, 
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DWRDAM_UNetV4, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            p=p,
            base_channels=base_channels
        )
        # 编码器
        self.encoder1 = DoubleConv(in_channels,     base_channels)
        self.a1 = AMSFN(base_channels)
        self.m1 = MDAM(base_channels)
        self.a2 = AMSFN(base_channels*2)
        self.m2 = MDAM(base_channels*2)
        self.a3 = AMSFN(base_channels*4)
        self.m3 = MDAM(base_channels*4)

    def forward(self, x):
        x1 = self.encoder1(x)
        x1 = self.encoder_dropout1(x1)

        x2 = self.down(x1)    
        a1 = self.a1(x2) 
        m1 = self.m1(x2)
        x2 = torch.cat([a1, m1], dim=1)
        x2 = self.encoder_dropout2(x2)                                 

        x3 = self.down(x2)
        a2 = self.a2(x3) 
        m2 = self.m2(x3)
        x3 = torch.cat([a2, m2], dim=1)                                  
        x3 = self.encoder_dropout3(x3)

        x4 = self.down(x3)
        a3 = self.a3(x4)
        m3 = self.m3(x4)
        x4 = torch.cat([a3, m3], dim=1)
        x4 = self.encoder_dropout4(x4)

        x5 = self.down(x4)
        x5 = self.center_conv(x5)                           # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x5)

        d4 = self.up(x)
        d4 = torch.cat([d4, x4], dim=1)                     # [b, 256, 32, 32]
        x = self.decoder1(d4)
        x = self.decoder_dropout1(x)   

        d3 = self.up(x)
        d3 = torch.cat([d3, x3], dim=1)                     # [b, 128, 64, 64]
        x = self.decoder2(d3)
        x = self.decoder_dropout2(x)

        d2 = self.up(x)
        d2 = torch.cat([d2, x2], dim=1)                     # [b, 64, 128, 128]
        x = self.decoder3(d2)

        d1 = self.up(x)
        d1 = torch.cat([d1, x1], dim=1)                     # [b, 64, 256, 256]
        x = self.decoder4(d1)
        logits = self.out_conv(x)                           # [1, c, 256, 256]
        return logits

class DWRDAM_UNetV5(DWRDAM_UNet): # 不同深度解码器直接上采样还原至原图分辨率，最后将不同深度的特征图进行CAT最后分类  
    def __init__(self, 
                 in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DWRDAM_UNetV5, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            p=p,
            base_channels=base_channels
        )
        self.up  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = AMSFNV2(in_channels,        base_channels)
        self.encoder2 = AMSFNV2(base_channels,      base_channels*2)
        self.encoder3 = AMSFNV2(base_channels*2,    base_channels*4) 
        self.encoder4 = AMSFNV2(base_channels*4,    base_channels*8) 
        # 解码器
        self.decoder1 = DWDoubleConv(base_channels * 16 ,   base_channels)
        self.decoder2 = DWDoubleConv(base_channels * 4,     base_channels)
        self.decoder3 = DWDoubleConv(base_channels * 2,     base_channels)
        self.decoder4 = DWDoubleConv(base_channels,         base_channels)

        self.out_conv = nn.Sequential(
                        nn.Conv2d(base_channels * 4, base_channels,  kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_channels, n_classes, kernel_size=1))

    def forward(self, x):
        x1 = self.encoder1(x)
        m1 = self.att1(x1)
        x1 = self.encoder_dropout1(x1)

        x2 = self.down(x1)    
        x2 = self.encoder2(x2) 
        m2 = self.att2(x2)                              # [b, 64, 128, 128]
        x2 = self.encoder_dropout2(x2)

        x3 = self.down(x2)
        x3 = self.encoder3(x3)                              # [b, 128, 64, 64]
        m3 = self.att3(x3)
        x3 = self.encoder_dropout3(x3)

        x4 = self.down(x3)
        x4 = self.encoder4(x4)                              # [b, 256, 32, 32]
        m4 = self.att4(x4)
        x4 = self.encoder_dropout4(x4)

        x5 = self.down(x4)
        x5 = self.center_conv(x5)                           # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x5)

        d4 = self.up(x)
        d4 = torch.cat([d4, m4], dim=1)                     # [b, 256, 32, 32]
        x = self.decoder1(d4)
        x1 = self.up1(x)
        x = self.decoder_dropout1(x)   

        x2 = self.decoder2(m3)
        x2 = self.up2(x2)
        x = self.decoder_dropout2(x)

        x3 = self.decoder3(m2)
        x3 = self.up3(x3)

        x4 = self.decoder4(m1)
        x = torch.cat([x1, x2, x3, x4], dim=1)            # [b, 64*4, 256, 256]
        logits = self.out_conv(x)                           # [1, c, 256, 256]
        return logits

class MAUNet(DWRDAM_UNet):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super().__init__(in_channels, n_classes, p, base_channels)
        # 编码器
        self.encoder1 = AMSFNV4(in_channels,      base_channels, )
        self.encoder2 = AMSFNV4(base_channels,    base_channels*2)
        self.encoder3 = AMSFNV4(base_channels*2,  base_channels*4)
        self.encoder4 = AMSFNV4(base_channels*4,  base_channels*8)  

        # 解码器
        self.decoder1 = Att_AWConv(base_channels * 16 , base_channels * 4)
        self.decoder2 = Att_AWConv(base_channels * 8,   base_channels * 2)
        self.decoder3 = Att_AWConv(base_channels * 4,   base_channels)
        self.decoder4 = Att_AWConv(base_channels * 2,   base_channels)
    def forward(self, x):
        return super().forward(x)
    
if __name__ == "__main__":
    from utils import * 
    model1 = DWRDAM_UNetV4(in_channels=3, n_classes=4, p=0)
    # model2 = RDAM_UNet(in_channels=3, n_classes=4, p=0)
    # x = torch.randn(1, 3, 256, 256)
    # y1 = model1(x)
    print(summary(model1, (1, 3, 256, 256)))
    calculate_computation(model1, input_size=(3, 256, 256), device='cuda')
    # ========================================
    # Input size: (3, 256, 256)
    # FLOPs: 3.80 GFLOPs
    # MACs: 1.90 GMACs
    # Params: 2.18 M
    # ========================================
    # calculate_computation(model2, input_size=(3, 256, 256), device='cuda')
    # ========================================
    # Input size: (3, 256, 256)
    # FLOPs: 16.62 GFLOPs
    # MACs: 8.31 GMACs
    # Params: 9.38 M
    # ========================================

else:
    from models.utils import *
