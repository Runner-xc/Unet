import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter  

class TransUNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 ):
        super(TransUNet, self).__init__()
        self.in_channels = in_channels

        self.encoder1 = AIConv2d(in_channels, base_channels) 
        self.encoder2 = AIConv2d(base_channels, base_channels*2)
        self.encoder3 = AIConv2d(base_channels*2, base_channels*4)
        self.encoder4 = AIConv2d(base_channels*4, base_channels*8)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        # dropout
        self.encoder_dropout = nn.Dropout2d(p=p)                            # 编码器更高dropout
        self.decoder_dropout = nn.Dropout2d(p=p-0.1 if p-0.1>0 else 0.0)    # 解码器较低dropout
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        
        self.decoder1 = AIConv2d(base_channels * 16, base_channels * 4)
        self.decoder2 = AIConv2d(base_channels * 8, base_channels * 2)
        self.decoder3 = AIConv2d(base_channels * 4, base_channels)
        self.decoder4 = AIConv2d(base_channels * 2, base_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = OutConv(base_channels, n_classes)
        
    def forward(self, x):
        x1 = self.encoder1(x)             # [1, 32, 320, 320]

        x2 = self.down(x1)
        x2 = self.encoder2(x2)             # [1, 64, 160, 160]
        x2 = self.encoder_dropout(x2)

        x3 = self.down(x2)        
        x3 = self.encoder3(x3)             # [1, 128, 80, 80]
        x3 = self.encoder_dropout(x3)

        x4 = self.down(x3)
        x4 = self.encoder4(x4)             # [1, 256, 40, 40]
        x4 = self.encoder_dropout(x4)

        x5 = self.down(x4)
        x5 = self.encoder_dropout(x5)
                   
        x = self.center_conv(x5)        # [1, 256, 20, 20]
        x = self.bottleneck_dropout(x)
        
        x = self.up(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder1(x)             # [1, 512, 40, 40]
        x = self.decoder_dropout(x)

        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder2(x)             # [1, 128, 80, 80]
        x = self.decoder_dropout(x)

        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder3(x)             # [1, 64, 160, 160]
        x = self.decoder_dropout(x)

        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder4(x)             # [1, 32, 320, 320]
        x = self.decoder_dropout(x)

        logits = self.out_conv(x)       # [1, c, 320, 320]       
        return logits 