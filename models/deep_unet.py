import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter 
  
class DeepSV_DW_UNet(nn.Module):
    """
    deep_supervised UNet with MambaLayer and Attention DWConv
    """
    def __init__(self, in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DeepSV_DW_UNet, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 编码器
        self.encoder1 = Att_AWBlock(in_channels,     base_channels)
        self.encoder2 = Att_AWBlock(base_channels,   base_channels*2)
        self.encoder3 = Att_AWBlock(base_channels*2, base_channels*4)
        self.encoder4 = Att_AWBlock(base_channels*4, base_channels*8)
        # encoder_dropout
        self.encoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.encoder_dropout2 = nn.Dropout2d(p=p*0.5 if p!=0 else 0)
        self.encoder_dropout3 = nn.Dropout2d(p=p*0.7 if p!=0 else 0)
        self.encoder_dropout4 = nn.Dropout2d(p=p*0.9 if p!=0 else 0)

        # bottleneck
        self.center_conv = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=1),
            MambaLayer(dim=base_channels*4, d_state=256, d_conv=16),
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=1))
        
        self.bottleneck_dropout = nn.Dropout2d(p=p if p!=0.0 else 0.0)
        # 解码器
        self.decoder1 = Att_AWBlock(base_channels * 16 , base_channels * 4) 
        self.decoder2 = Att_AWBlock(base_channels * 8 ,  base_channels * 2)
        self.decoder3 = Att_AWBlock(base_channels * 4 ,  base_channels)
        self.decoder4 = Att_AWBlock(base_channels * 2,   base_channels)
        # decoder_dropout
        self.decoder_dropout1 = nn.Dropout2d(p=p*0.3 if p!=0 else 0)
        self.decoder_dropout2 = nn.Dropout2d(p=p*0.2 if p!=0 else 0)
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.convd2 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.convd3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.convd4 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
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
        d4 = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.decoder_dropout1(d4)

        x = self.up(x)                      # [1, 128, 80, 80]
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        d3 = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.decoder_dropout2(d3)

        x = self.up(x)                      # [1, 64, 160, 160]
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        d2 = self.decoder3(x)                # [1, 32, 160, 160]

        x = self.up(d2)                      # [1, 32, 160, 160]
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        
        logits = self.out_conv(x)           # [1, c, 320, 320]

        if self.training:  #或 self.training
            d2 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(d2)
            d2 = self.convd2(d2)
            d3 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(d3)
            d3 = self.convd3(d3)
            d4 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(d4)
            d4 = self.convd4(d4)
            return {"deep_supervision": (logits, d2, d3, d4)}
        else:
            return logits

class DeepSV_DW_UNetV2(DeepSV_DW_UNet):
    """
    deep_supervised UNet with MambaLayer and Attention DWConv
    """
    def __init__(self, in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 ):
        super(DeepSV_DW_UNetV2, self).__init__(in_channels, num_classes, p, base_channels)
        # 编码器
        self.encoder1 = AMSFNV4(in_channels,     base_channels)
        self.encoder2 = AMSFNV4(base_channels,   base_channels*2)
        self.encoder3 = AMSFNV4(base_channels*2, base_channels*4)
        self.encoder4 = AMSFNV4(base_channels*4, base_channels*8)
    def forward(self, x):
        return super(DeepSV_DW_UNetV2, self).forward(x)
        
if __name__ == '__main__':
    from utils import *  
    model = DeepSV_DW_UNetV2(in_channels=3, num_classes=4, p=0)
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
 
        