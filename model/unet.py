"""
unet
"""
import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter   
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        
        self.up1 = Up(base_channels * 8, base_channels * 4, bilinear=bilinear)
        self.up2 = Up(base_channels * 4, base_channels * 2, bilinear=bilinear)
        self.up3 = Up(base_channels * 2, base_channels,     bilinear=bilinear)
        self.up4 = Up(base_channels,     base_channels,     bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)             # [1, 64, 320, 320]
        x2 = self.down1(x1)             # [1, 128, 160, 160]
        x2 = self.encoder_dropout(x2)
        x3 = self.down2(x2)             # [1, 256, 80, 80]
        x3 = self.encoder_dropout(x3)
        x4 = self.down3(x3)             # [1, 512, 40, 40]
        x4 = self.encoder_dropout(x4)
        x5 = self.pool(x4)              # [1, 512, 20, 20]
                   
        x = self.center_conv(x5)        # [1, 512, 20, 20]
        x = self.bottleneck_dropout(x)
           
        x = self.up1(x, x4)             # [1, 256, 40, 40]
        x = self.decoder_dropout(x)
        x = self.up2(x, x3)             # [1, 128, 80, 80]
        x = self.decoder_dropout(x)
        x = self.up3(x, x2)             # [1, 64, 160, 160]
        x = self.decoder_dropout(x)
        x = self.up4(x, x1)             # [1, 64, 320, 320]
        x = self.decoder_dropout(x)
        logits = self.out_conv(x)       # [1, c, 320, 320]       
        return logits 
          
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class ResD_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(ResD_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.down1 = ResD_Down(base_channels, base_channels*2)  
        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.down3 = ResD_Down(base_channels*4, base_channels*8)

        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16) 

        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.up2 = ResD_Up(base_channels * 4 , base_channels * 2, bilinear=bilinear)
        self.up3 = ResD_Up(base_channels * 2 , base_channels,     bilinear=bilinear)
        self.up4 = Up(base_channels,      base_channels,     bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)             # [1, 64, 320, 320]

        x2 = self.down1(x1)             # [1, 128, 160, 160]
        x2 = self.encoder_dropout(x2)

        x3 = self.down2(x2)             # [1, 256, 80, 80]
        x3 = self.encoder_dropout(x3)

        x4 = self.down3(x3)             # [1, 512, 40, 40]
        x4 = self.encoder_dropout(x4)

        x5 = self.pool(x4)              # [1, 512, 20, 20]           
        x = self.center_conv(x5)        # [1, 512, 20, 20]
        x = self.bottleneck_dropout(x)
           
        x = self.up1(x, x4)             # [1, 256, 40, 40]
        x = self.decoder_dropout(x)

        x = self.up2(x, x3)             # [1, 128, 80, 80]
        x = self.decoder_dropout(x)

        x = self.up3(x, x2)             # [1, 64, 160, 160]
        x = self.decoder_dropout(x)

        x = self.up4(x, x1)             # [1, 64, 320, 320]
        x = self.decoder_dropout(x)

        logits = self.out_conv(x)       # [1, c, 320, 320]       
        return logits
    
            
if __name__ == '__main__':
    from utils.attention import EMA
    from utils.modules import *   
    model = UNet(in_channels=3, n_classes=4, p=0.25)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(3,320,320)
    summary(model)
else:
    from model.utils.attention import *
    from model.utils.modules import * 
        
