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
        self.down4 = Down(base_channels*8, base_channels*16)
        
        self.dropout = nn.Dropout2d(p=p)
        self.center_conv = DoubleConv(base_channels*16, base_channels*16)
        
        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear=bilinear)
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
           
        x = self.up1(x, x4)        # [1, 256, 40, 40]
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
        self.inconv = DoubleConv(in_channels, base_channels)
        # 编码器
        self.down1 = DWDown(base_channels, base_channels*2)
        self.msaf1 = EMAF(base_channels*2)
        self.acpn1 = ACPN(base_channels*2)
        self.conv1 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=1)

        self.down2 = DWResD_Down(base_channels*2, base_channels*4)
        self.msaf2 = EMAF(base_channels*4)
        self.acpn2 = ACPN(base_channels*4)
        self.conv2 = nn.Conv2d(base_channels*8, base_channels*4, kernel_size=1)

        self.down3 = DWResD_Down(base_channels*4, base_channels*8)
        self.msaf3 = EMAF(base_channels*8)
        # self.acpn3 = ACPN(base_channels*8)
        self.conv3 = nn.Conv2d(base_channels*16, base_channels*8, kernel_size=1)

        self.down4 = DWResD_Down(base_channels*8, base_channels*16)
        self.msaf4 = EMAF(base_channels*16)
        # self.acpn4 = ACPN(base_channels*16)
        self.conv4 = nn.Conv2d(base_channels*32, base_channels*16, kernel_size=1)
        
        self.dropout = nn.Dropout2d(p=p)
        self.center_conv = DoubleConv(base_channels*16, base_channels*16)
        
        # 解码器
        self.up1 = DWResD_Up(base_channels * 16 , base_channels * 8, bilinear=bilinear)
        self.msaf5 = EMAF(base_channels*8)

        self.up2 = DWResD_Up(base_channels * 8, base_channels * 4 , bilinear=bilinear)
        self.msaf6 = EMAF(base_channels*4)

        self.up3 = DWResD_Up(base_channels * 4, base_channels * 2, bilinear=bilinear)
        self.acpn7 = ACPN(base_channels*2)
        self.msaf7 = EMAF(base_channels*2)
        self.conv7 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=1)

        self.up4 = DWUp(base_channels * 2, base_channels, bilinear=bilinear)
        self.msaf8 = EMAF(base_channels)
        self.acpn8 = ACPN(base_channels)
        self.conv8 = nn.Conv2d(base_channels*2, base_channels, kernel_size=1)

        # 输出层
        self.out_conv = OutConv(base_channels, n_classes)
        
        # self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        # self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        # self.conv3 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):
        e1 = self.inconv(x)                             # [b, 32, 256, 256]
        # down, up = self.down_and_up(e1)

        e2 = self.down1(e1)                             # [b, 64, 128, 128]
        m = self.msaf1(e2)
        a = self.acpn1(e2)
        e2 = torch.cat([m, a], dim=1)
        e2 = self.conv1(e2)
        x = self.dropout(e2)
                  
        e3 = self.down2(x)                              # [b, 128, 64, 64]
        m = self.msaf2(e3)
        a = self.acpn2(e3)
        e3 = torch.cat([m, a], dim=1)
        e3 = self.conv2(e3)
        x = self.dropout(e3)
        
        e4 = self.down3(x)                              # [b, 256, 32, 32]
        m = self.msaf3(e4)
        # a = self.acpn3(e4)
        # e4 = torch.cat([m, a], dim=1)
        x = self.dropout(m)

        e5 = self.down4(x)                              # [b, 512, 16, 16]
        m = self.msaf4(e5)
        # a = self.acpn4(e5)
        # e5 = torch.cat([m, a], dim=1)
        x = self.dropout(m)
           
        c5 = self.center_conv(x)                                    # [b, 512, 16, 16]
           
        d4 = self.up1(c5, e4)                                       # [b, 256, 32, 32]
        d4 = self.msaf5(d4)
        x = self.dropout(d4)
        
        # c5_up = F.interpolate(c5, scale_factor=2, mode='bilinear')  # [b, 512, 32, 32]
        # c5_up = self.conv1(c5_up)
        d3 = self.up2(d4, e3)                           # [b, 128, 80, 80]
        d3 = self.msaf6(d3)
        x = self.dropout(d3)

        # c5_up = F.interpolate(c5_up, scale_factor=2, mode='bilinear')
        # c5_up = self.conv2(c5_up)
        # d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear')
        # d4_up = self.conv2(d4_up)
        d2 = self.up3(d3, e2)              # [b, 64, 160, 160]
        m = self.msaf7(d2)
        a = self.acpn7(d2)
        d2 = torch.cat([m, a], dim=1)
        d2 = self.conv7(d2)
        x = self.dropout(d2)

        # c5_up = F.interpolate(c5_up, scale_factor=2, mode='bilinear')
        # c5_up = self.conv3(c5_up)
        # d4_up = F.interpolate(d4_up, scale_factor=2, mode='bilinear')
        # d4_up = self.conv3(d4_up)
        # d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear')
        # d3_up = self.conv3(d3_up)
        d1 = self.up4(d2, e1)                           # [1, 64, 320, 320]
        m = self.msaf8(d1)
        a = self.acpn8(d1)
        d1 = torch.cat([m, a], dim=1)
        d1 = self.conv8(d1)
        x = self.dropout(d1)
        logits = self.out_conv(x)                       # [1, c, 320, 320]
        
        return logits

    # def down_and_up(self, x):
    #     down = []
    #     for i in range(3):
    #         weight1 = torch.randn((self.base_channels*2**(i+1), self.base_channels*2**i, 1, 1)).to('cuda')
    #         x = F.conv2d(x, weight1)
    #         x = F.max_pool2d(x, 2, 2)
    #         down.append(x)

    #     up = []
    #     for i in range(3):
    #         weight2 = torch.randn((self.base_channels*2**(3-i-1), self.base_channels*2**(3-i), 1, 1)).to('cuda')
    #         x = F.conv2d(x, weight2)
    #         x = F.interpolate(x, scale_factor=2, mode='bilinear')
    #         up.append(x)
    #     return down, up

class Res_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(Res_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels*2)  
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = SE_Down(base_channels*4, base_channels*8)  # 残差下采样
        factor = 2 if bilinear else 1
        self.down4 = SE_Down(base_channels*8, base_channels*16 // factor)
        
        self.dropout = nn.Dropout2d(p=p)

        self.center_conv = DoubleConv(base_channels*16 // factor, base_channels*16) 

        self.up1 = SE_Up(base_channels * 16, base_channels * 8 // factor, bilinear)  # 残差上采样
        self.up2 = SE_Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
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
        
        x = self.up1(x, x4)        # [1, 256, 32, 32]
        x = self.dropout(x)         # dropout层
        x = self.up2(x, x3)         # [1, 128, 64, 64]
        x = self.dropout(x)         # dropout层
        x = self.up3(x, x2)         # [1, 64, 128, 128]
        x = self.dropout(x)             
        x = self.up4(x, x1)         # [1, 32, 256, 256]
        x = self.dropout(x)         # dropout层
        logits = self.out_conv(x)   # [1, c, 256, 256]       
        return logits
    
class RDHAM_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(RDHAM_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.down1 = D_Down(base_channels, base_channels*2)  
        self.down2 = D_Down(base_channels*2, base_channels*4)
        self.down3 = Res_Down(base_channels*4, base_channels*8)  # 残差下采样
        factor = 2 if bilinear else 1
        self.down4 = ResD_Down(base_channels*8, base_channels*16 // factor)
        
        self.dropout = nn.Dropout2d(p=p)
        
        # 残差注意力模块
        self.ham = Res_HAM(base_channels*16 // factor)
        self.center_conv = DoubleConv(base_channels*16 // factor, base_channels*16) 

        self.up1 = D_Up(base_channels * 16, base_channels * 8 // factor, bilinear)  # 残差上采样
        self.up2 = D_Up(base_channels * 8, base_channels * 4 // factor, bilinear)
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
        
        x = self.ham(x5)            # [1, 512, 16, 16]
        x = self.center_conv(x5)    # [1, 512, 16, 16]
        
        x = self.up1(x, x4)        # [1, 256, 32, 32]
        x = self.dropout(x)         # dropout层
        x = self.up2(x, x3)         # [1, 128, 64, 64]
        x = self.dropout(x)         # dropout层
        x = self.up3(x, x2)         # [1, 64, 128, 128]
        x = self.dropout(x)             
        x = self.up4(x, x1)         # [1, 32, 256, 256]
        x = self.dropout(x)         # dropout层
        logits = self.out_conv(x)   # [1, c, 256, 256]       
        return logits

class SE_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True,
                 flag = False
                 ):
        super(SE_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.flag = flag

        # CAB 模块
        self.causality_map_block = CausalityMapBlock()
        self.causality_factors_extractor = CausalityFactorsExtractor()

        self.inconv = DoubleConv(in_channels, base_channels)
        self.psp = PSPModule(base_channels)
        self.conv = DoubleConv(base_channels*2, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        # 加入SE注意力机制
        self.down3 = SE_Down(base_channels*4, base_channels*8)
        factor = 2 if bilinear else 1
        self.down4 = SE_Down(base_channels*8, base_channels*16 // factor)
        self.dropout = nn.Dropout2d(p=p)
        
        self.center_conv = DoubleConv(base_channels*16 // factor, base_channels*16)
        # 加入SE注意力机制
        self.up1 = SE_Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = SE_Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear) 
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)         # [1, 32, 256, 256]
        ## 加入CAB模块
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
           
        x = self.up1(x, x4)        # [1, 256, 32, 32]
        x = self.dropout(x)         
        x = self.up2(x, x3)         # [1, 128, 64, 64]
        x = self.dropout(x)         
        x = self.up3(x, x2)         # [1, 64, 128, 128]
        x = self.dropout(x)         
        # 增加特征金字塔池化
        if self.flag:
            x = self.up4(x, x_psp)      # [1, 32, 256, 256]
        else:
            x = self.up4(x, x1)         # [1, 32, 256, 256]
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
    from attention import EMA
    from modules import *   
    model = MSAF_UNet(in_channels=3, n_classes=4, p=0.25)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(3,320,320)
    summary(model)
else:
    from .attention import *
    from .modules import * 
        