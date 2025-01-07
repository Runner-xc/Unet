import torch.nn as nn
import torch
import torch.nn.functional as F
from .modules import *
from torchinfo import summary
from .resnet_backbone import *

class ConvBNReluLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1, padding=None, name=None):
        """
        in_channels: 输入数据的通道数
        out_channels: 输出数据的通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        groups: 二维卷积层的组数
        dilation: 空洞大小
        padding: 填充大小
        """
        super(ConvBNReluLayer, self).__init__()
        if padding is None:
            padding = (kernel_size-1)//2
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    padding=padding,
                                    groups=groups,
                                    dilation=dilation,
                                    )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = +self.bn(x)
        x = self.relu(x)
        return x

class BottleneckBlock(nn.Module):
    expansion = 4  # 最后的输出的通道数会变成4倍
    def __init__(self, in_channels, out_channels, stride=1, shortcut=True, dilation=1, padding=None, name=None):
        """
        shortcut: 最开始的输入和输出是否能进行直接相加的操作, 能=True, 不能=False（会进行1x1卷积操作进行维度匹配）。
        """
        super(BottleneckBlock, self).__init__()
        # 3次 (卷积、批归一化、ReLU) 操作
        self.conv0 = ConvBNReluLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv1 = ConvBNReluLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = ConvBNReluLayer(in_channels=out_channels, out_channels=out_channels*4, kernel_size=1, stride=1)
        
        if not shortcut:
            # 不能直接相加时, 进行维度匹配
            self.short = ConvBNReluLayer(in_channels=in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride)
        
        self.shortcut = shortcut
        self.num_channel_out = out_channels * 4  # 最后的输出通道数变成4倍
    
    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = torch.add(short, conv2)  # 元素相加
        y = torch.nn.functional.relu(y)  # ReLU激活
        return y
    
def resnet50(**kwargs):
    res50 = ResNet(Bottleneck, [3, 4, 6, 3],**kwargs) 
    return res50
    
class PSPNet(nn.Module):
    def __init__(self, num_classes, dropout_p, use_aux=True):
        super(PSPNet, self).__init__()
        res = resnet50()  # 生成backbone
        self.use_aux = use_aux  # 是否用辅助分类器，True则用
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv = res.conv1
        self.bn = res.bn1
        self.relu = res.relu
        self.pool2d_max = res.maxpool
        
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.conv2 = nn.Conv2d(1024, 128, kernel_size=1)
        self.layer4 = res.layer4
        self.conv3 = nn.Conv2d(2048, 256, kernel_size=1)
        num_channels = 256
        self.pspmodule = PSPModule(num_channels, [1, 2, 3, 6])
        
        # cls：2048*2->512->num_classes，经过PSPModule，通道数会翻倍
        self.classifier = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(num_channels // 4, num_classes, kernel_size=1)
        )
        
        # aux
        self.aux = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(num_channels // 4, num_classes, kernel_size=1)
        )
        
        
    def forward(self, inputs):
        input_size = inputs.shape[2:]  # H, W
        # inputs: [3, 1, 1]
        x = self.conv(inputs)  # [64, 1/2, 1/2]
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool2d_max(x)

        x = self.layer1(x)  # [256, 1/4, 1/4]
        x = self.dropout(x)

        x = self.layer2(x)  # [512, 1/8, 1/8]
        x = self.dropout(x)

        aux_x = self.layer3(x)  # [1024, 1/16, 1/16]
        x = self.dropout(aux_x)
        aux_x = self.conv2(x)

        x = self.layer4(x)  # [2048, 1/32, 1/32]
        x = self.dropout(x)
        x = self.conv3(x)

        out = self.pspmodule(x)  # [4096, 1/32, 1/32]
        out = self.dropout(out)

        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)  # [4096, 1, 1]
        heatmap = self.classifier(out)  # [num_classes, 1, 1]
        if self.use_aux:
            aux = self.aux(aux_x)  # [num_classes, 1, 1]
            aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
            return heatmap, aux
        return heatmap
    
if __name__ == '__main__':
    net = PSPNet(num_classes=3, use_aux=True, dropout_p=0.5)
    x = torch.randn(2, 3, 256, 256)
    heatmap, aux = net(x)
    summary(model=net)
    print(heatmap.shape)
    print(aux.shape)
    