"""
SED_unet
"""
import torch
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F

'''-------------一、SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.cbr1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbr2 = nn.Sequential(self.conv2, self.bn2, self.relu)
        
    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        return x
    
class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(TripleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.cbr1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        
        self.dconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.cbr2 = nn.Sequential(self.dconv2, self.bn2, self.relu)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.cbr3 = nn.Sequential(self.conv3, self.bn3, self.relu)
        
    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        return x
"----------------------------------------------下采样--------------------------------------------------------------------------"       
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.triple_conv = TripleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.triple_conv(x)
        return x

class SE_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SE_Down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.triple_conv = TripleConv(in_channels, out_channels)
        self.se = SE_Block(out_channels, ratio=int(0.25*out_channels))
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.triple_conv(x)
        x = self.se(x) 
        return x
"----------------------------------------------上采样--------------------------------------------------------------------------"         
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = TripleConv(in_channels, out_channels, in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = TripleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class SE_Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(SE_Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = TripleConv(in_channels, out_channels, in_channels//2)
            self.se = SE_Block(out_channels, ratio=int(0.25*out_channels))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = TripleConv(in_channels, out_channels)
            self.se = SE_Block(out_channels, ratio=int(0.25*out_channels))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x

"----------------------------------------------输出--------------------------------------------------------------------------"       
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
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
        
        self.inconv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        
        # 加入SE注意力机制
        self.down3 = SE_Down(base_channels*4, base_channels*8)
        factor = 2 if bilinear else 1
        self.down4 = SE_Down(base_channels*8, base_channels*16 // factor)
        
        self.dropout = nn.Dropout2d(p=p)
        
        # 加入SE注意力机制
        self.up1 = SE_Up(base_channels * 16, base_channels * 8 // factor, bilinear)
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
        x5 = self.down4(x4)         # [1, 256, 16, 16]
           
        x = self.up1(x5, x4)        # [1, 128, 32, 32]
        x = self.up2(x, x3)         # [1, 64, 64, 64]
        x = self.up3(x, x2)         # [1, 32, 128, 128]
        x = self.up4(x, x1)         # [1, 32, 256, 256]
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
    summary(model, (1,3,256,256))