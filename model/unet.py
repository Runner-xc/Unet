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
        
        self.dropout = nn.Dropout2d(p=p)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        
        self.up1 = Up(base_channels * 8, base_channels * 4, bilinear=bilinear)
        self.up2 = Up(base_channels * 4, base_channels * 2, bilinear=bilinear)
        self.up3 = Up(base_channels * 2, base_channels,     bilinear=bilinear)
        self.up4 = Up(base_channels,     base_channels,     bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)         # [1, 64, 320, 320]
        x2 = self.down1(x1)         # [1, 128, 160, 160]
        x2 = self.dropout(x2)       
        x3 = self.down2(x2)         # [1, 256, 80, 80]
        x3 = self.dropout(x3)       
        x4 = self.down3(x3)         # [1, 512, 40, 40]
        x4 = self.dropout(x4)       
        x5 = self.pool(x4)         # [1, 512, 20, 20]
           
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
        self.acpn0 = ACPN(base_channels)
        
        # 编码器 
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.attention_dropout = nn.Dropout2d(p=p-0.2 if p-0.2>0 else 0.0)  # 注意力模块后dropout

        # self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
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
        a1 = self.acpn0(e1)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]
        x1 = self.encoder_dropout(a1)
        # m1 = self.attention_dropout(m1)

        e2 = self.down1(x1)                              # [b, 64, 128, 128]
        a2 = self.acpn1(e2)
        m2 = self.msaf2(e2)
        x2 = self.encoder_dropout(a2)
        # m2 = self.attention_dropout(m2)
        
        e3 = self.down2(x2)                              # [b, 128, 64, 64]
        a3 = self.acpn2(e3)
        m3 = self.msaf3(e3)
        x3 = self.encoder_dropout(a3)
        # m3 = self.attention_dropout(m3)

        e4 = self.down3(x3)                              # [b, 256, 32, 32]
        a4 = self.acpn3(e4)
        m4 = self.msaf4(e4)
        x4 = self.encoder_dropout(a4)
        # m4 = self.attention_dropout(m4)
        
        x5 = self.pool(x4)
        # x = self.dense_aspp(x)                         # [b, 512, 16, 16]   
        c5 = self.center_conv(x5)                        # [b, 512, 16, 16]

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

class A_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(A_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels
        # 编码器
        self.inconv = DoubleConv(in_channels, base_channels)
        self.acpn0 = ACPN(base_channels)   
        self.down1 = ResD_Down(base_channels, base_channels*2)
        self.acpn1 = ACPN(base_channels*2)
        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.acpn2 = ACPN(base_channels*4)      
        self.down3 = ResD_Down(base_channels*4, base_channels*8)
        self.acpn3 = ACPN(base_channels*8)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.attention_dropout = nn.Dropout2d(p=p-0.2 if p-0.2>0 else 0.0)  # 注意力模块后dropout
        # bottleneck
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
        a1 = self.acpn0(e1)                             # [b, 32, 256, 256]
        x1 = self.encoder_dropout(a1)
        e2 = self.down1(x1)                              # [b, 64, 128, 128]
        a2 = self.acpn1(e2)
        x2 = self.encoder_dropout(a2)
        e3 = self.down2(x2)                              # [b, 128, 64, 64]
        a3 = self.acpn2(e3)
        x3 = self.encoder_dropout(a3)
        e4 = self.down3(x3)                              # [b, 256, 32, 32]
        a4 = self.acpn3(e4)
        x4 = self.encoder_dropout(a4)
        
        x5 = self.pool(x4)
        # x = self.dense_aspp(x)                         # [b, 512, 16, 16]   
        c5 = self.center_conv(x5)                        # [b, 512, 16, 16]

        d4 = self.up1(c5, x4)                            # [b, 256, 32, 32]
        d4 = self.acpn5(d4)
        x  = self.decoder_dropout(d4)   
        d3 = self.up2(x, x3)                             # [b, 128, 64, 64]
        d3 = self.acpn6(d3)
        x  = self.decoder_dropout(d3)
        d2 = self.up3(x, x2)                             # [b, 64, 128, 128]
        d2 = self.acpn7(d2)
        x  = self.decoder_dropout(d2)
        d1 = self.up4(x, x1)                             # [1, 64, 256, 256]
        d1 = self.acpn8(d1)
        x  = self.decoder_dropout(d1)
        logits = self.out_conv(x)                        # [1, c, 256, 256]
        return logits

class M_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(M_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels
        # 编码器 
        self.inconv = DoubleConv(in_channels, base_channels)
        self.msaf1 = EMAF(base_channels) 
        self.down1 = ResD_Down(base_channels, base_channels*2)
        self.msaf2 = EMAF(base_channels*2)
        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.msaf3 = EMAF(base_channels*4)      
        self.down3 = ResD_Down(base_channels*4, base_channels*8)
        self.msaf4 = EMAF(base_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.attention_dropout = nn.Dropout2d(p=p-0.2 if p-0.2>0 else 0.0)  # 注意力模块后dropout
        # bottleneck
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        # 解码器
        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.up2 = ResD_Up(base_channels * 4, base_channels * 2 , bilinear=bilinear)   
        self.up3 = ResD_Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.up4 = Up(base_channels, base_channels, bilinear=bilinear)
        # 输出层
        self.out_conv = OutConv(base_channels, n_classes)

    def forward(self, x):
        e1 = self.inconv(x)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]
        x1 = self.encoder_dropout(e1)
        # m1 = self.attention_dropout(m1)
        e2 = self.down1(x1)                              # [b, 64, 128, 128]
        m2 = self.msaf2(e2)
        x2 = self.encoder_dropout(e2)
        # m2 = self.attention_dropout(m2)
        e3 = self.down2(x2)                              # [b, 128, 64, 64]
        m3 = self.msaf3(e3)
        x3 = self.encoder_dropout(e3)
        # m3 = self.attention_dropout(m3)
        e4 = self.down3(x3)                              # [b, 256, 32, 32]
        m4 = self.msaf4(e4)
        x4 = self.encoder_dropout(e4)
        # m4 = self.attention_dropout(m4)
        
        x5 = self.pool(x4)  
        c5 = self.center_conv(x5)                        # [b, 512, 16, 16]

        d4 = self.up1(c5, m4)                            # [b, 256, 32, 32]
        x = self.decoder_dropout(d4)   
        d3 = self.up2(x, m3)                             # [b, 128, 64, 64]
        x = self.decoder_dropout(d3)
        d2 = self.up3(x, m2)                             # [b, 64, 128, 128]
        x = self.decoder_dropout(d2)
        d1 = self.up4(x, m1)                             # [1, 64, 256, 256]
        x = self.decoder_dropout(d1)
        logits = self.out_conv(x)                        # [1, c, 256, 256]
        return logits

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
        
        self.inconv = ResDConv(in_channels, base_channels)
        self.down1 = ResD_Down(base_channels, base_channels*2)  
        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.down3 = ResD_Down(base_channels*4, base_channels*8)

        self.dropout = nn.Dropout2d(p=p)
        self.encoder_dropout = nn.Dropout2d(p=0.3)  # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=0.2)  # 解码器较低dropout
        self.attention_dropout = nn.Dropout2d(p=0.1) # 注意力模块后dropout

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16) 

        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.up2 = ResD_Up(base_channels * 4 , base_channels * 2, bilinear=bilinear)
        self.up3 = ResD_Up(base_channels * 2 , base_channels,     bilinear=bilinear)
        self.up4 = ResD_Up(base_channels,      base_channels,     bilinear=bilinear)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.inconv(x)             # [1, 64, 320, 320]
        x1 = self.encoder_dropout(x1)

        x2 = self.down1(x1)             # [1, 128, 160, 160]
        x2 = self.encoder_dropout(x2)

        x3 = self.down2(x2)             # [1, 256, 80, 80]
        x3 = self.encoder_dropout(x3)

        x4 = self.down3(x3)             # [1, 512, 40, 40]
        x4 = self.encoder_dropout(x4)

        x5 = self.pool(x4)              # [1, 512, 20, 20]           
        x = self.center_conv(x5)        # [1, 512, 20, 20]
           
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