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

def split_class(img_pred, img_mask):
    """
    img_pred: 预测值 (batch, 4, h, w)
    img_mask: 标签值 (batch, h, w)
    """
    
    class_0 = img_pred[:, 0, ...]
    class_1 = img_pred[:, 1, ...]
    class_2 = img_pred[:, 2, ...]
    class_3 = img_pred[:, 3, ...]
    pre_class_list = [class_0, class_1, class_2, class_3]

    # 确保img_mask是长整型张量
    mask_0 = img_mask[:, 0, ...]
    mask_1 = img_mask[:, 1, ...]
    mask_2 = img_mask[:, 2, ...]
    mask_3 = img_mask[:, 3, ...]
    mask_class_list = [mask_0, mask_1, mask_2, mask_3]

    return pre_class_list, mask_class_list

"""
Cross Entropy Loss
"""
class CrossEntropyLoss():
    def __init__(self, **kwargs):
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

    def __call__(self, img_pred: Tensor, img_mask: Tensor) -> Tensor:
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        loss_dict = {}
        num_classes = self.num_classes
        class_names = self.class_names
        
        # img_mask: (b, h, w) -> (b, c, h, w)
        img_mask = img_mask.to(torch.int64)
        img_mask = F.one_hot(img_mask, num_classes).permute(0, 3, 1, 2).float()
        pred_class_list, mask_class_list = split_class(img_pred, img_mask)

        # 计算出每个类别的损失
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            loss = nn.CrossEntropyLoss()(pred_class, mask_class)
            loss_dict[class_name] = loss

        OM_loss = loss_dict['Organic matter']
        OP_loss = loss_dict['Organic pores']
        IOP_loss = loss_dict['Inorganic pores']
        mean_loss = sum(loss_dict.values()) / len(loss_dict)
        
        return OM_loss, OP_loss, IOP_loss, mean_loss
    
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

    def __call__(self, img_pred, img_mask):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        loss_dict = {}
        class_names = self.class_names
        num_classes = self.num_classes
        tensor_one = torch.tensor(1)

        # img_mask: (b, h, w) -> (b, c, h, w)
        img_mask = img_mask.to(torch.int64)
        img_mask = F.one_hot(img_mask, num_classes + 1).permute(0, 3, 1, 2).float()  
        pred_class_list, mask_class_list = split_class(img_pred, img_mask)

        # 遍历计算每个类别的损失
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            intersection = (pred_class * mask_class).sum()
            union = pred_class.sum() + mask_class.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            loss_dict[class_name] = tensor_one - dice

        # 损失值    
        OM_loss = loss_dict['Organic matter']
        OP_loss = loss_dict['Organic pores']
        IOP_loss = loss_dict['Inorganic pores']
        mean_loss = (sum(loss_dict.values()) - loss_dict['Background']) / len(loss_dict)
        

        return OM_loss, OP_loss, IOP_loss, mean_loss
    
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

    def __call__(self,img_pred, img_mask):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        loss_dict = {}
        class_names = self.class_names
        num_classes = self.num_classes
        crossentropy = nn.CrossEntropyLoss()

        # img_mask: (b, h, w) -> (b, c, h, w)
        img_mask = img_mask.to(torch.int64)
        img_mask = img_mask.squeeze(1)
        img_mask = F.one_hot(img_mask, num_classes + 1).permute(0, 3, 1, 2).float()
        pred_class_list, mask_class_list = split_class(img_pred, img_mask)

        # 遍历计算每个类别的损失
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            ce_loss = crossentropy(pred_class, mask_class)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * ((1-pt)**self.gamma) * ce_loss
            loss_dict[class_name] = focal_loss

        # 损失值    
        OM_loss = loss_dict['Organic matter']
        OP_loss = loss_dict['Organic pores']
        IOP_loss = loss_dict['Inorganic pores']
        mean_loss = sum(loss_dict.values()) / len(loss_dict)

        return OM_loss, OP_loss, IOP_loss, mean_loss

            
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
