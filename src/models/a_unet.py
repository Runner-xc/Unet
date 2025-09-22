import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter
 
class A_UNetV2(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
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
        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)

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
                num_classes,
                p,
                base_channels=32,
                ):
        super(A_UNetV3, self).__init__(
                 in_channels,
                 num_classes,
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
                num_classes,
                p,
                base_channels=32,
                ):
        super().__init__(
                 in_channels,
                 num_classes,
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
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(A_UNetV5, self).__init__(in_channels, num_classes, p, base_channels)
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
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(A_UNetV6, self).__init__(in_channels, num_classes, p, base_channels)
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
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(Mamba_AUNet, self).__init__(in_channels, num_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=128, d_conv=8),
            nn.Conv2d(base_channels*4,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)
class Mamba_AUNetV3(A_UNetV4):
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(Mamba_AUNetV3, self).__init__(in_channels, num_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*2, kernel_size=1),
            MambaLayer(dim=base_channels*2, d_state=128, d_conv=8),
            nn.Conv2d(base_channels*2,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)
    
class Mamba_AUNetV4(A_UNetV4):
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(Mamba_AUNetV4, self).__init__(in_channels, num_classes, p, base_channels)
        # bottleneck使用MambaLayer
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8,base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=48, d_conv=8),
            nn.Conv2d(base_channels*4,base_channels*8, kernel_size=1))
      
    def forward(self, x):
        return super().forward(x)

class Mamba_AUNetV5(A_UNetV4):
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(Mamba_AUNetV5, self).__init__(in_channels, num_classes, p, base_channels)
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
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(Mamba_AUNetV6, self).__init__(in_channels, num_classes, p, base_channels)
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
    def __init__(self, in_channels, num_classes, p, base_channels=32):
        super(Mamba_AUNetV2, self).__init__(in_channels, num_classes, p, base_channels)
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
    
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
        return l1_lambda * l1_loss + l2_lambda * l2_loss
if __name__ == '__main__':
    from utils import *    
    model = Mamba_AUNetV2(in_channels=3, num_classes=4, p=0)
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