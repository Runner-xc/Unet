import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
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
        self.c2 = nn.Sequential(self.conv2, self.bn2, self.relu)
        
    def forward(self, x1):
        x = self.cbr1(x1)
        x = self.c2(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DWConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x

class RDModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2 
        self.DalitConv = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        res = x
        res = self.conv(res)
        x = self.DalitConv(x)
        x = torch.add(x, res)
        x = self.relu(x)
        return x

class AMSFN(nn.Module):  
    def __init__(self, in_channels, mid_channels=None):
        super(AMSFN, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        # 3×3
        self.cbrp1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbrp2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbrp3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sgm = nn.Sigmoid()
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        )

    def forward(self, x):
        inputs = x
        b, c, h, w = inputs.size()
        c1 = self.maxpool(inputs)
        s1 = self.avg_pool(c1)

        c2 = self.cbrp1(inputs)
        s2 = self.avg_pool(c2)

        c3 = self.cbrp2(inputs)
        s3 = self.avg_pool(c3)

        c4 = self.cbrp3(inputs)
        s4 = self.avg_pool(c4)

        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.mlp(out)
        weights = self.sgm(out)

        # 将权重作用到各个卷积结果上并相加
        c1_weighted = c1 * weights[:, 0:c, ...]
        c2_weighted = c2 * weights[:, c:c*2 , ...]
        c3_weighted = c3 * weights[:, c*2:c*3, ...]
        c4_weighted = c4 * weights[:, c*3:, ...]

        output = c1_weighted + c2_weighted + c3_weighted + c4_weighted
        output = self.final_conv(output)
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.resd = RDModule(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        self.amsfn = AMSFN(out_channels, mid_channels=mid_channels)
        
    def forward(self, x):
        x = self.resd(x)
        x = self.amsfn(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.resd = RDModule(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        self.amsfn = AMSFN(out_channels, mid_channels=mid_channels)
        
    def forward(self, x):
        x = self.resd(x)
        x = self.amsfn(x)
        return x
    
class MDAM(nn.Module):
    def __init__(self, channels, w=0.5,factor=32):
        super(MDAM, self).__init__()
        self.group = factor
        self.w = w
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=1)
        self.averagePooling = nn.AdaptiveAvgPool2d((1,1))
        self.Pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.Pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.groupNorm = nn.GroupNorm(channels // self.group, channels//self.group)
        self.conv1x1 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        group_x = inputs.reshape(b*self.group, -1, h, w)
        x_h = self.Pool_h(group_x)  # 高度方向池化
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2)  # 宽度方向池化

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # 拼接之后卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)       # 拆分

        # H W
        x = self.groupNorm(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())          # 高度的注意力
        x = self.softmax(self.averagePooling(x))

        # SE
        y = self.conv3x3(group_x) # 通过 3x3卷积层
        y = self.gap(y)
        y = self.fc(y)

        weights = self.w*x + (1-self.w)*y

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
       
class RDAM_UNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 p, 
                 w,
                 base_channels=32,
                 ):
        super(RDAM_UNet, self).__init__()
        self.down = self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = Encoder(in_channels, base_channels, base_channels)
        self.msaf1 = MDAM(base_channels, w=w) 

        self.encoder2 = Encoder(base_channels, base_channels*2)      
        self.msaf2 = MDAM(base_channels*2, w=w)
        self.encoder_dropout2 = nn.Dropout2d(p=p-0.2 if p!=0 else 0)  

        self.encoder3= Encoder(base_channels*2, base_channels*4)
        self.msaf3 = MDAM(base_channels*4, w=w)
        self.encoder_dropout3 = nn.Dropout2d(p=p-0.1 if p!=0 else 0)  

        self.encoder4 = Encoder(base_channels*4, base_channels*8)
        self.msaf4 = MDAM(base_channels*8, w=w)
        self.encoder_dropout4 = nn.Dropout2d(p=p)                           

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        # 解码器
        self.decoder4 = Decoder(base_channels * 16, base_channels * 4)
        self.decoder_dropout4 = nn.Dropout2d(0.15 if p!=0 else 0)   

        self.decoder3 = Decoder(base_channels * 8, base_channels * 2)  
        self.decoder_dropout3 = nn.Dropout2d(0.1 if p!=0 else 0) 

        self.decoder2 = Decoder(base_channels * 4, base_channels)
        self.decoder_dropout2 = nn.Dropout2d(0.05 if p!=0 else 0) 

        self.decoder1 = Decoder(base_channels * 2, base_channels)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.down(e1)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]

        e2 = self.encoder2(x)                              # [b, 64, 128, 128]
        x = self.down(e2)
        m2 = self.msaf2(e2)
        x = self.encoder_dropout2(x)
        
        e3 = self.encoder3(x)                              # [b, 128, 64, 64]
        x = self.down(e3)
        m3 = self.msaf3(e3)
        x = self.encoder_dropout3(x)

        e4 = self.encoder4(x)                              # [b, 256, 32, 32]
        x = self.down(e4)
        m4 = self.msaf4(e4)
        x = self.encoder_dropout4(x)
        
        x = self.center_conv(x)                        # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x)

        x = self.up(x)
        x = torch.cat([x, m4], dim=1)                            # [b, 256, 32, 32]
        d4 = self.decoder4(x)
        x = self.decoder_dropout4(d4)   

        x = self.up(x)                             # [b, 128, 64, 64]
        x = torch.cat([x, m3], dim=1) 
        d3 = self.decoder3(x)
        x = self.decoder_dropout3(d3)

        x = self.up(x)                             # [b, 64, 128, 128]
        x = torch.cat([x, m2], dim=1) 
        d2 = self.decoder2(x)
        x = self.decoder_dropout2(d2)

        x = self.up(x)                             # [1, 64, 256, 256]
        x = torch.cat([x, m1], dim=1) 
        d1 = self.decoder1(x)
        logits = self.out_conv(d1)                        # [1, c, 256, 256]
        
        return logits
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class M_UNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 p, 
                 w=0.7,
                 base_channels=32,
                 bilinear=True
                 ):
        super(M_UNet, self).__init__()
        self.down = self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.msaf1 = MDAM(base_channels, w=w) 

        self.encoder2 = DoubleConv(base_channels, base_channels*2)      
        self.msaf2 = MDAM(base_channels*2, w=w)
        self.encoder_dropout2 = nn.Dropout2d(p=p-0.2 if p!=0 else 0)  

        self.encoder3= DoubleConv(base_channels*2, base_channels*4)
        self.msaf3 = MDAM(base_channels*4, w=w)
        self.encoder_dropout3 = nn.Dropout2d(p=p-0.1 if p!=0 else 0)  

        self.encoder4 = DoubleConv(base_channels*4, base_channels*8)
        self.msaf4 = MDAM(base_channels*8, w=w)
        self.encoder_dropout4 = nn.Dropout2d(p=p)                           

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        # 解码器
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 4)
        self.decoder_dropout4 = nn.Dropout2d(0.15 if p!=0 else 0)   

        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 2)  
        self.decoder_dropout3 = nn.Dropout2d(0.1 if p!=0 else 0) 

        self.decoder2 = DoubleConv(base_channels * 4, base_channels)
        self.decoder_dropout2 = nn.Dropout2d(0.05 if p!=0 else 0) 

        self.decoder1 = DoubleConv(base_channels * 2, base_channels)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.down(e1)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]

        e2 = self.encoder2(x)                              # [b, 64, 128, 128]
        x = self.down(e2)
        m2 = self.msaf2(e2)
        x = self.encoder_dropout2(x)
        
        e3 = self.encoder3(x)                              # [b, 128, 64, 64]
        x = self.down(e3)
        m3 = self.msaf3(e3)
        x = self.encoder_dropout3(x)

        e4 = self.encoder4(x)                              # [b, 256, 32, 32]
        x = self.down(e4)
        m4 = self.msaf4(e4)
        x = self.encoder_dropout4(x)
        
        x = self.center_conv(x)                        # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x)

        x = self.up(x)
        x = torch.cat([x, m4], dim=1)                            # [b, 256, 32, 32]
        d4 = self.decoder4(x)
        x = self.decoder_dropout4(d4)   

        x = self.up(x)                             # [b, 128, 64, 64]
        x = torch.cat([x, m3], dim=1) 
        d3 = self.decoder3(x)
        x = self.decoder_dropout3(d3)

        x = self.up(x)                             # [b, 64, 128, 128]
        x = torch.cat([x, m2], dim=1) 
        d2 = self.decoder2(x)
        x = self.decoder_dropout2(d2)

        x = self.up(x)                             # [1, 64, 256, 256]
        x = torch.cat([x, m1], dim=1) 
        d1 = self.decoder1(x)
        logits = self.out_conv(d1)                        # [1, c, 256, 256]
        return logits
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss
    
class A_UNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(A_UNet, self).__init__()
        self.down = self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = Encoder(in_channels, base_channels, base_channels)

        self.encoder2 = Encoder(base_channels, base_channels*2)      
        self.encoder_dropout2 = nn.Dropout2d(p=p-0.2 if p!=0 else 0)  

        self.encoder3= Encoder(base_channels*2, base_channels*4)
        self.encoder_dropout3 = nn.Dropout2d(p=p-0.1 if p!=0 else 0)  

        self.encoder4 = Encoder(base_channels*4, base_channels*8)
        self.encoder_dropout4 = nn.Dropout2d(p=p)                           

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        # 解码器
        self.decoder4 = Decoder(base_channels * 16, base_channels * 4)
        self.decoder_dropout4 = nn.Dropout2d(0.15 if p!=0 else 0)   

        self.decoder3 = Decoder(base_channels * 8, base_channels * 2)  
        self.decoder_dropout3 = nn.Dropout2d(0.1 if p!=0 else 0) 

        self.decoder2 = Decoder(base_channels * 4, base_channels)
        self.decoder_dropout2 = nn.Dropout2d(0.05 if p!=0 else 0) 

        self.decoder1 = Decoder(base_channels * 2, base_channels)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.down(e1)

        e2 = self.encoder2(x)                              # [b, 64, 128, 128]
        x = self.down(e2)
        x = self.encoder_dropout2(x)
        
        e3 = self.encoder3(x)                              # [b, 128, 64, 64]
        x = self.down(e3)
        x = self.encoder_dropout3(x)

        e4 = self.encoder4(x)                              # [b, 256, 32, 32]
        x = self.down(e4)
        x = self.encoder_dropout4(x)
        
        x = self.center_conv(x)                        # [b, 512, 16, 16]
        x = self.bottleneck_dropout(x)

        x = self.up(x)
        x = torch.cat([x, e4], dim=1)                            # [b, 256, 32, 32]
        d4 = self.decoder4(x)
        x = self.decoder_dropout4(d4)   

        x = self.up(x)                             # [b, 128, 64, 64]
        x = torch.cat([x, e3], dim=1) 
        d3 = self.decoder3(x)
        x = self.decoder_dropout3(d3)

        x = self.up(x)                             # [b, 64, 128, 128]
        x = torch.cat([x, e2], dim=1) 
        d2 = self.decoder2(x)
        x = self.decoder_dropout2(d2)

        x = self.up(x)                             # [1, 64, 256, 256]
        x = torch.cat([x, e1], dim=1) 
        d1 = self.decoder1(x)
        logits = self.out_conv(d1)                        # [1, c, 256, 256]
        return logits
    def elastic_net(self, l1_lambda, l2_lambda):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.pow(param, 2).sum()
            
        return l1_lambda * l1_loss + l2_lambda * l2_loss

if __name__ == "__main__":
    model = A_UNet(in_channels=3, num_classes=4, p=0)
    summary(model, (1, 3, 256, 256))
