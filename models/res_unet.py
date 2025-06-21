import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from .utils import *
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Encoder, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1,0), groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,3), stride=1, padding=(0,1), groups=in_channels)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True))
        
        self.conv3x3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 + x2
        x = self.conv3(x)
        x = self.conv3x3(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Decoder, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1,0), groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,3), stride=1, padding=(0,1), groups=in_channels)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True))
        
        self.conv3x3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 + x2
        x = self.conv3(x)
        x = self.conv3x3(x)
        return x
    
class SegModel(nn.Module):
    def __init__(self, num_classes):
        super(SegModel, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = Encoder(3, 64)
        self.encoder2 = Encoder(64, 128)

        self.encoder3 = Encoder(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = Decoder(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = Decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.down(x1)
        x2 = self.encoder2(x2)
        c = self.down(x2)
        c = self.encoder3(c)
        
        x3 = self.up1(c)
        x3 = torch.cat((x3, x2), dim=1) 
        d1 = self.decoder1(x3)

        x4 = self.up2(d1)
        x4 = torch.cat((x4, x1), dim=1)
        d2 = self.decoder2(x4)

        c = self.up2(self.up1(c))
        d1 = self.up2(d1)
        out = self.final_conv(d2) + self.final_conv(d1) + self.final_conv(c)
        return out

class RES50_UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(RES50_UNet, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder1 = resnet.layer1  # 256
        self.encoder2 = resnet.layer2  # 512
        self.encoder3 = resnet.layer3  # 1024
        self.encoder4 = resnet.layer4  # 2048

        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = DoubleConv(2048, 512)
        self.decoder2 = DoubleConv(1024, 256)
        self.decoder3 = DoubleConv(512, 64)
        self.final_conv = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        c = self.bottleneck(x5)

        d1 = self.up(c)
        d1 = torch.cat((d1, x4), dim=1)
        d1 = self.decoder1(d1)

        d2 = self.up(d1)
        d2 = torch.cat((d2, x3), dim=1)
        d2 = self.decoder2(d2)

        d3 = self.up(d2)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.decoder3(d3)
        out = self.final_conv(d3)
        return out
    
class RES18_UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(RES18_UNet, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = DoubleConv(512, 128)
        self.decoder2 = DoubleConv(256, 64)
        self.decoder3 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        c = self.bottleneck(x5)

        d1 = self.up(c)
        d1 = torch.cat((d1, x4), dim=1)
        d1 = self.decoder1(d1)

        d2 = self.up(d1)
        d2 = torch.cat((d2, x3), dim=1)
        d2 = self.decoder2(d2)

        d3 = self.up(d2)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.decoder3(d3)
        out = self.final_conv(d3)
        return out

if __name__ == "__main__":
    model = SegModel(num_classes=10)  # Example with 10 classes
    x = torch.randn(1, 3, 256, 256)  # Example input
    output = model(x)
    summary(model, input_size=(1, 3, 256, 256), device='cpu')
    print(output.shape)  # Should be (1, num_classes, 256, 256)