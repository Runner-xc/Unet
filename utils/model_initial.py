import torch
import torch.nn as nn

# kaiming初始化
def kaiming_initial(model, init_gain=0.02):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, init_gain)
            nn.init.constant_(m.bias, 0)

# 正交初始化
def orthogonal_initial(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.orthogonal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, init_gain)
            nn.init.constant_(m.bias, 0)

# xavier初始化
def xavier_initial(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, init_gain)
            nn.init.constant_(m.bias, 0)