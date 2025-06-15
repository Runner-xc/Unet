import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter
 
class A_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(A_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 编码器
        self.encoder1 = DoubleConv(in_channels,       base_channels, )
        self.encoder2 = DoubleConv(base_channels,     base_channels*2)
        self.encoder3 = DoubleConv(base_channels*2,   base_channels*4)
        self.encoder4 = DoubleConv(base_channels*4,   base_channels*8)
        # 多尺度融合
        self.acpn0 = AMSFN(base_channels)
        self.acpn1 = AMSFN(base_channels*2) 
        self.acpn2 = AMSFN(base_channels*4) 
        self.acpn3 = AMSFN(base_channels*8)
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
        self.decoder1 = DoubleConv(base_channels * 16 ,   base_channels * 4)
        self.decoder2 = DoubleConv(base_channels * 8,     base_channels * 2)
        self.decoder3 = DoubleConv(base_channels * 4,     base_channels)
        self.decoder4 = DoubleConv(base_channels * 2,     base_channels)
        # 多尺度融合
        self.acpn5 = AMSFN(base_channels*4)
        self.acpn6 = AMSFN(base_channels*2) 
        self.acpn7 = AMSFN(base_channels) 
        self.acpn8 = AMSFN(base_channels)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)

        # 输出层
        self.out_conv = nn.Conv2d(base_channels, n_classes, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        x1 = self.acpn0(e1)                                 
        x2 = self.down(x1)
        x2 = self.encoder_dropout1(x2)                                

        e2 = self.encoder2(x2)                              # [b, 64, 128, 128]
        a2 = self.acpn1(e2)
        x3 = self.down(a2)
        x3 = self.encoder_dropout2(x3)
        
        e3 = self.encoder3(x3)                              # [b, 128, 64, 64]
        a3 = self.acpn2(e3)
        x4 = self.down(a3)
        x4 = self.encoder_dropout3(x4)

        e4 = self.encoder4(x4)                              # [b, 256, 32, 32]
        a4 = self.acpn3(e4)
        x5 = self.down(a4)
        x5 = self.encoder_dropout4(x5)
                            
        c5 = self.center_conv(x5)                           # [b, 512, 16, 16]
        c5 = self.bottleneck_dropout(c5)

        d4 = self.up(c5)
        d4 = torch.cat([d4, e4], dim=1)                     # [b, 256, 32, 32]
        d4 = self.decoder1(d4)
        d4 = self.acpn5(d4)
        x = self.decoder_dropout1(d4)   

        d3 = self.up(x)
        d3 = torch.cat([d3, e3], dim=1)                      # [b, 128, 64, 64]
        d3 = self.decoder2(d3)
        d3 = self.acpn6(d3)
        x = self.decoder_dropout2(d3)

        d2 = self.up(x)
        d2 = torch.cat([d2, e2], dim=1)                      # [b, 64, 128, 128]
        d2 = self.decoder3(d2)
        d2 = self.acpn7(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)                     # [b, 64, 256, 256]
        d1 = self.decoder4(d1)
        x = self.acpn8(d1)
        logits = self.out_conv(x)                           # [1, c, 256, 256]
        
        return logits

class A_UNetV2(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(A_UNetV2, self).__init__( 
        )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = AMSFNV3(in_channels,      base_channels, )
        self.encoder2 = AMSFNV3(base_channels,    base_channels*2)
        self.encoder3 = AMSFNV3(base_channels*2,  base_channels*4)
        self.encoder4 = AMSFNV3(base_channels*4,  base_channels*8)
        # bottleneck
        self.center_conv = DWDoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        # 解码器
        self.decoder1 = AWConv(base_channels * 16 , base_channels * 4)
        self.decoder2 = AWConv(base_channels * 8,   base_channels * 2)
        self.decoder3 = AWConv(base_channels * 4,   base_channels)
        self.decoder4 = AWConv(base_channels * 2,   base_channels)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, n_classes, 1)

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
        x = torch.cat([x, x4], dim=1)       # [1, 256+256, 40, 40]
        x = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up(x)                      # [1, 128, 80, 80]
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        x = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up(x)                      # [1, 64, 160, 160]
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        x = self.decoder3(x)                # [1, 32, 160, 160]

        x = self.up(x)                      # [1, 32, 160, 160]
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        
        logits = self.out_conv(x)           # [1, c, 320, 320]       
        return logits 
class A_UNetV3(A_UNetV2):
    def __init__(self, 
                in_channels,
                n_classes,
                p,
                base_channels=32,
                ):
        super(A_UNetV3, self).__init__(
                 in_channels,
                 n_classes,
                 p, 
                 base_channels,
        )
        # 解码器
        self.decoder1 = Att_AWConv(base_channels * 16 , base_channels * 4)
        self.decoder2 = Att_AWConv(base_channels * 8,   base_channels * 2)
        self.decoder3 = Att_AWConv(base_channels * 4,   base_channels)
        self.decoder4 = Att_AWConv(base_channels * 2,   base_channels)
    
    def forward(self, x):
        return super(A_UNetV3, self).forward(x)

class A_UNetV4(A_UNetV3):
    def __init__(self, 
                in_channels,
                n_classes,
                p,
                base_channels=32,
                ):
        super().__init__(
                 in_channels,
                 n_classes,
                 p, 
                 base_channels,
        )
        # 编码器
        self.encoder1 = AMSFNV4(in_channels,      base_channels, )
        self.encoder2 = AMSFNV4(base_channels,    base_channels*2)
        self.encoder3 = AMSFNV4(base_channels*2,  base_channels*4)
        self.encoder4 = AMSFNV4(base_channels*4,  base_channels*8)

    def forward(self, x):
        return super().forward(x)
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class A_UNetV5(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(A_UNetV5, self).__init__(in_channels, n_classes, p, base_channels)
        self.center_conv = Att_AWConv(base_channels*8, base_channels*8)
    def forward(self, x):
        return super().forward(x)
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss

class A_UNetV6(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(A_UNetV6, self).__init__(in_channels, n_classes, p, base_channels)
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            Att_AWBlock(base_channels*8, base_channels*8))

    def forward(self, x):
        return super().forward(x)
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss

class Mamba_AUNet(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(Mamba_AUNet, self).__init__(in_channels, n_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=128, d_conv=8),
            nn.Conv2d(base_channels*4,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)
class Mamba_AUNetV3(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(Mamba_AUNetV3, self).__init__(in_channels, n_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*2, kernel_size=1),
            MambaLayer(dim=base_channels*2, d_state=128, d_conv=8),
            nn.Conv2d(base_channels*2,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)
    
class Mamba_AUNetV4(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(Mamba_AUNetV4, self).__init__(in_channels, n_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=48, d_conv=8),
            nn.Conv2d(base_channels*4,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)

class Mamba_AUNetV5(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(Mamba_AUNetV5, self).__init__(in_channels, n_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=256, d_conv=8),
            nn.Conv2d(base_channels*4,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class Mamba_AUNetV6(A_UNetV4):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(Mamba_AUNetV6, self).__init__(in_channels, n_classes, p, base_channels)
        # 调整MambaLayer参数
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=256, d_conv=16),
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=1))

    def forward(self, x):
        return super().forward(x)
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        return l1_lambda * l1_loss + l2_lambda * l2_loss

class Mamba_AUNetV2(Mamba_AUNet):
    def __init__(self, in_channels, n_classes, p, base_channels=32):
        super(Mamba_AUNetV2, self).__init__(in_channels, n_classes, p, base_channels)
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*2, kernel_size=1),
            MambaLayer(dim=base_channels*2, d_state=64, d_conv=8),
            nn.Conv2d(base_channels*2,base_channels*8, kernel_size=1))
        
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
        x5 = self.encoder_dropout4(x4)
       
        x = self.center_conv(x5)            # [1, 256, 20, 20]
        x = self.bottleneck_dropout(x)
        
        x = torch.cat([x, x4], dim=1)       # [1, 256+256, 40, 40]
        x = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.decoder_dropout1(x)

        x = self.up(x)                      # [1, 128, 80, 80]
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        x = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.decoder_dropout2(x)

        x = self.up(x)                      # [1, 64, 160, 160]
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        x = self.decoder3(x)                # [1, 32, 160, 160]

        x = self.up(x)                      # [1, 32, 160, 160]
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        
        logits = self.out_conv(x)           # [1, c, 320, 320]       
        return logits 
if __name__ == '__main__':
    from utils import *    
    model = A_UNetV4(in_channels=3, n_classes=4, p=0)
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
    from models.utils import *