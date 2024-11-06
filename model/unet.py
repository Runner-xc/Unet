"""
unet
"""
import torch
from torchinfo import summary
import torch.nn as nn
from .modules import *      
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
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels*8, base_channels*16 // factor)
        
        self.dropout = nn.Dropout2d(p=p)
        self.center_conv = DoubleConv(base_channels*16 // factor, base_channels*16)
        
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)         # [1, 64, 320, 320]
        x2 = self.down1(x1)         # [1, 128, 160, 160]
        x2 = self.dropout(x2)       
        x3 = self.down2(x2)         # [1, 256, 80, 80]
        x3 = self.dropout(x3)       
        x4 = self.down3(x3)         # [1, 512, 40, 40]
        x4 = self.dropout(x4)       
        x5 = self.down4(x4)         # [1, 512, 20, 20]
           
        x = self.center_conv(x5)    # [1, 512, 20, 20]
           
        x = self.up1(x5, x4)        # [1, 256, 40, 40]
        x = self.dropout(x)
        x = self.up2(x, x3)         # [1, 128, 80, 80]
        x = self.dropout(x)         
        x = self.up3(x, x2)         # [1, 64, 160, 160]
        x = self.dropout(x)
        x = self.up4(x, x1)         # [1, 64, 320, 320]
        x = self.dropout(x)
        logits = self.out_conv(x)   # [1, c, 320, 320]
        
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
        self.down2 = Res_Down(base_channels*2, base_channels*4)
        self.down3 = Res_Down(base_channels*4, base_channels*8)  # 残差下采样
        factor = 2 if bilinear else 1
        self.down4 = Res_Down(base_channels*8, base_channels*16 // factor)
        
        self.dropout = nn.Dropout2d(p=p)
        self.center_conv = DoubleConv(base_channels*16 // factor, base_channels*16) 

        self.up1 = Res_Up(base_channels * 16, base_channels * 8 // factor, bilinear)  # 残差上采样
        self.up2 = Res_Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Res_Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = ResD_Up(base_channels * 2, base_channels, bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)         # [1, 32, 256, 256]
        x2 = self.down1(x1)         # [1, 64, 128, 128]
        x2 = self.dropout(x2)       # dropout层
        x3 = self.down2(x2)         # [1, 128, 64, 64]
        x3 = self.dropout(x3)       # dropout层
        x4 = self.down3(x3)         # [1, 256, 32, 32]
        x4 = self.dropout(x4)       # dropout层
        x5 = self.down4(x4)         # [1, 512, 16, 16]
        x5 = self.dropout(x5)       # dropout层
        
        x = self.center_conv(x5)    # [1, 512, 16, 16]
        
        x = self.up1(x5, x4)        # [1, 256, 32, 32]
        x = self.dropout(x)         # dropout层
        x = self.up2(x, x3)         # [1, 128, 64, 64]
        x = self.dropout(x)         # dropout层
        x = self.up3(x, x2)         # [1, 64, 128, 128]
        x = self.dropout(x)             
        x = self.up4(x, x1)         # [1, 32, 256, 256]
        x = self.dropout(x)         # dropout层
        logits = self.out_conv(x)   # [1, c, 256, 256]       
        return logits
        
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()           
        return l1_lambda * l1_loss + l2_lambda * l2_loss

class SED_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(SED_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # CAB 模块
        self.causality_map_block = CausalityMapBlock()
        self.causality_factors_extractor = CausalityFactorsExtractor()
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.psp = PSPModule(base_channels)
        self.conv = DoubleConv(base_channels*2, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        # 加入SE注意力机制
        self.down3 = Down(base_channels*4, base_channels*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels*8, base_channels*16 // factor)
        self.dropout = nn.Dropout2d(p=p)
        
        self.center_conv = DoubleConv(base_channels*16 // factor, base_channels*16)
        # 加入SE注意力机制
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear) 
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)         # [1, 32, 256, 256]
        # 加入CAB模块
        # x1 = self.causality_factors_extractor(x1, self.causality_map_block(x1))
        x1 = self.dropout(x1) 
              
        x_psp = self.psp(x1)        # [1, 64, 256, 256]
        x_psp = self.dropout(x_psp)
        x_psp = self.conv(x_psp)     # [1, 32, 256, 256]

        x2 = self.down1(x1)         # [1, 64, 128, 128]
        x2 = self.dropout(x2)
               
        x3 = self.down2(x2)         # [1, 128, 64, 64]
        x3 = self.dropout(x3)
               
        x4 = self.down3(x3)         # [1, 256, 32, 32]
        x4 = self.dropout(x4)
        
        x5 = self.down4(x4)         # [1, 256, 16, 16]
        x5 = self.dropout(x5)
        # x5 = self.causality_factors_extractor(x4, self.causality_map_block(x4))

        x = self.center_conv(x5)    # [1, 512, 16, 16]
           
        x = self.up1(x5, x4)        # [1, 256, 32, 32]
        x = self.dropout(x)         
        x = self.up2(x, x3)         # [1, 128, 64, 64]
        x = self.dropout(x)         
        x = self.up3(x, x2)         # [1, 64, 128, 128]
        x = self.dropout(x)         
        # 增加特征金字塔池化
        x = self.up4(x, x_psp)      # [1, 32, 256, 256]
        x = self.dropout(x)
        # x = self.causality_factors_extractor(x, self.causality_map_block(x))          
        logits = self.out_conv(x)   # [1, c, 256, 256]        
        return logits
        
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()            
        return l1_lambda * l1_loss + l2_lambda * l2_loss
            
if __name__ == '__main__':
    model = SED_UNet(in_channels=3, n_classes=4, p=0.25)
    x = torch.randn(1,3,320,320)
    # output = model(x)
    # print(model)
    summary(model, (1,3,224,224))