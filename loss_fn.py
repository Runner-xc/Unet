"""
损失函数:Dice Loss、BoundaryLoss、Focal Loss、DistanceBasedLoss、Tversky Loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from torch import einsum
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


"""
Cross Entropy Loss
"""
class CrossEntropyLoss():
    def __init__(self):
        self.class_names = [
                            'Background',
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Background':0,
            'Organic matter':1,
            'Organic pores':2,
            'Inorganic pores':3
        }

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        
        # 计算loss
        target = targets
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        ce = nn.CrossEntropyLoss(label_smoothing=1e-7, reduction='none')    # 不进行缩减会返回（batch, h, w）的loss值
       
        # celoss期望的logits是(b, c, h, w), targets是(b, h, w)
        loss = ce(logits, targets)   # [8, 320, 320]
        total_loss = loss.mean()
        
        # # 初始化一个数组来存储每个类别的损失
        # class_losses = torch.zeros(num_classes)

        # # 计算每个类别的平均损失
        # for i in range(num_classes):
        #     mask = (target == i)  # 创建一个 mask，其中类别 i 的位置为 True
        #     class_loss = loss[mask]  # 选择该类别的损失
        #     class_losses[i] = class_loss.mean()  # 计算该类别的平均损失

        # 记录每个类别的损失
        loss_dict = {}
        loss_dict['total_loss'] = total_loss
        # loss_dict['Background'] = class_losses[0]
        # loss_dict['Organic matter'] = class_losses[1]
        # loss_dict['Organic pores'] = class_losses[2]
        # loss_dict['Inorganic pores'] = class_losses[3]
        
        return loss_dict
    
class DiceLoss():
    
    def __init__(self, smooth=1e-5):
        """
        smooth: 平滑值
        """
        self.smooth = smooth
        self.class_names = [
                            'Background',
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Background':0,
            'Organic matter':1,
            'Organic pores':2,
            'Inorganic pores':3
        }
        self.num_classes = len(self.class_names)

    def __call__(self, logits, targets):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = self.num_classes
        tensor_one = torch.tensor(1)

        # logits argmax
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  
        
        # 计算总的损失
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = tensor_one - dice
        total_loss = loss.sum() / num_classes
        
        # 计算每个类别的损失
        loss_dict = {}
        loss_dict['total_loss'] = total_loss
        loss_dict['Background'] = loss[0]
        loss_dict['Organic matter'] = loss[1]
        loss_dict['Organic pores'] = loss[2]
        loss_dict['Inorganic pores'] = loss[3]

        return loss_dict
    
class Focal_Loss():
    """
    γ : 聚焦因子,用于控制损失的敏感度
    α : 平衡正负样本权重
    """
    def __init__(self,alpha=0.8,gamma=2):
        self.gamma=gamma
        self.alpha=alpha
        self.class_names = [
                            'Background',
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Background':0,
            'Organic matter':1,
            'Organic pores':2,
            'Inorganic pores':3
        }
        self.num_classes = len(self.class_names)

    def __call__(self,logits, targets):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """   
        num_classes = logits.shape[1]
        
        # 计算loss
        target = targets
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        ce = nn.CrossEntropyLoss(label_smoothing=1e-7, reduction='none')    # 不进行缩减会返回（batch, h, w）的loss值
       
        # celoss期望的logits是(b, c, h, w), targets是(b, h, w)
        ce_loss = ce(logits, targets)   # [8, 320, 320]
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1-pt)**self.gamma) * ce_loss
        total_loss = loss.mean()
        
        # # 初始化一个数组来存储每个类别的损失
        # class_losses = torch.zeros(num_classes)

        # # 计算每个类别的平均损失
        # for i in range(num_classes):
        #     mask = (target == i)  # 创建一个 mask，其中类别 i 的位置为 True
        #     class_loss = ce_loss[mask]  # 选择该类别的损失
        #     class_ce_loss = class_loss.mean()  # 计算该类别的平均损失
        #     pt = torch.exp(-class_ce_loss)
        #     class_loss = self.alpha * ((1-pt)**self.gamma) * class_ce_loss
        #     class_losses[i] = class_loss

        # 记录每个类别的损失
        loss_dict = {}
        loss_dict['total_loss'] = total_loss
        # loss_dict['Background'] = class_losses[0]
        # loss_dict['Organic matter'] = class_losses[1]
        # loss_dict['Organic pores'] = class_losses[2]
        # loss_dict['Inorganic pores'] = class_losses[3]

        return loss_dict

            
"""
boundary loss   # TODO: 待实验

"""



"""
基于距离的损失函数,计算预测分割与真实分割之间的平均表面距离。  # TODO: 待实验

参数:
    reduction (str): 指定损失的缩减方式,'mean' 或 'sum'。
"""
    

class TotalLoss(nn.Module):
    """
    loss_fn: 支持的损失函数类型，如 'dice'、'boundary'、'focal'、'distance_based'
    """
    def __init__(self, flag: bool = False, loss_fn: str = None):
        super().__init__()
        self.flag = flag
        self.dice_loss = DiceLoss()
        # self.boundary_loss = SurfaceLoss()
        self.focal_loss = Focal_Loss()
        # self.distance_based_loss = DistanceBasedLoss()  # 修改命名风格
        
        # 校验 loss_fn 参数的有效性
        if loss_fn is not None and loss_fn not in ['dice_loss', 'boundary_loss', 'focal_loss', 'distance_based_loss']:
            raise ValueError(f"Invalid loss function: {loss_fn}. Supported values are 'dice', 'boundary', 'focal', 'distance_based'.")
        self.loss_fn = loss_fn

    def __call__(self, pred, labels):
        # 判断是否使用多个损失函数组合
        if self.flag:
            return (self.dice_loss(pred, labels) + self.boundary_loss(pred, labels) + self.focal_loss(pred, labels))
        else:
            if self.loss_fn == 'dice':
                return self.dice_loss(pred, labels)
            # 'boundary'
            elif self.loss_fn == 'boundary':
                return self.boundary_loss(pred, labels)
            # 'focal'
            elif self.loss_fn == 'focal':
                return self.focal_loss(pred, labels)
            # 'distance_based'
            elif self.loss_fn == 'distance_based':
                return self.distance_based_loss(pred, labels)
            
            else:
                raise ValueError(f"Invalid loss function: {self.loss_fn}")
            

# if __name__ == '__main__':
#     loss_fn = TotalLoss(flag=True, loss_fn='boundary_loss')
#     x = torch.randn(1, 4, 256, 256)
#     y = torch.randint(0, 4, (1, 256, 256))
#     x = F.softmax(x, dim=1)
#     dice_loss = DiceLoss()
#     boundary_loss = SurfaceLoss()
#     focal_loss = Focal_Loss()
#     distance_based_loss = DistanceBasedLoss()
#     print(focal_loss(x, y))
#     print(boundary_loss(x, y))
#     print(dice_loss(x, y))
