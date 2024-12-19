"""
损失函数:Dice Loss、BoundaryLoss、Focal Loss、DistanceBasedLoss、Tversky Loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


"""
Cross Entropy Loss
"""
class CrossEntropyLoss():
    def __init__(self):
        self.class_names = [
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Organic matter':0,
            'Organic pores':1,
            'Inorganic pores':2
        }

    def __call__(self, logits, targets):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        
        # 计算loss
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        targets = targets[:, 1:, ...]
        ce = nn.CrossEntropyLoss(label_smoothing=1e-7, reduction='none', ignore_index=0)    # 不进行缩减会返回（batch, h, w）的loss值
       
        # celoss期望的logits是(b, c, h, w), targets是(b, h, w)
        loss_dict = {}
        names = ['Background', 'Organic matter', 'Organic pores', 'Inorganic pores']
        for i in range(num_classes):
            target = targets[:, i, ...]
            logit = logits[:, i, ...] 
            loss_dict[names[i]] = ce(logit, target)   # [b, 256, 256]  

        # 记录类别损失
        total_loss = torch.stack(list(loss_dict.values())).mean()
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
class DiceLoss():
    
    def __init__(self, smooth=1e-5):
        """
        smooth: 平滑值
        """
        self.smooth = smooth
        self.class_names = [
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Organic matter':0,
            'Organic pores':1,
            'Inorganic pores':2
        }
        self.num_classes = len(self.class_names)

    def __call__(self, logits, targets):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        tensor_one = torch.tensor(1)

        # logits argmax 
        logits = torch.softmax(logits, dim=1)  # 不能直接使用argmax，会丢失grad_fn,因为argmax不可导
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  
        logits = logits[:, 1:, ...]
        targets = targets[:, 1:, ...]
        # 计算总的损失
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        dice = (2 * intersection) / (union + self.smooth)
        loss = tensor_one - dice
        total_loss = loss.sum() / num_classes
        
        # 计算每个类别的损失
        loss_dict = {}
        loss_dict['total_loss'] = total_loss
        loss_dict['Organic matter'] = loss[0]
        loss_dict['Organic pores'] = loss[1]
        loss_dict['Inorganic pores'] = loss[2]

        return loss_dict
    
class Focal_Loss():
    """
    γ : 聚焦因子,用于控制损失的敏感度
    α : 平衡正负样本权重
    """
    def __init__(self,alpha=0.25, gamma=2):
        self.gamma=gamma
        self.alpha=alpha
        self.class_names = [
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Organic matter':0,
            'Organic pores':1,
            'Inorganic pores':2
        }
        self.num_classes = len(self.class_names)

    def __call__(self,logits, targets):
        """
        img_pred: 预测值 (batch, 3, h, w)
        img_mask: 标签值 (batch, h, w)
        """   
        num_classes = logits.shape[1]
        
        # 计算loss
        targets = targets.to(torch.int64)
        ce = nn.CrossEntropyLoss(label_smoothing=1e-7, reduction='none', ignore_index=0)    # 不进行缩减会返回（batch, h, w）的loss值
        
        # 计算 Focal Loss
        # celoss期望的logits是(b, c, h, w), targets是(b, h, w)
        ce_loss = ce(logits, targets)  # [b, 256, 256]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        # 计算每个类别的损失，忽略背景（索引0）
        loss_dict = {}
        total_loss = 0
        names = ['Organic matter', 'Organic pores', 'Inorganic pores']
        for i in range(1, self.num_classes+1):  # 从1开始，忽略背景
            class_loss = focal_loss[targets == i].mean()
            loss_dict[names[i-1]] = class_loss
            total_loss += class_loss
        
        # for i in range(1, num_classes):
        #     target = targets[:, i, ...]
        #        
        #     pt = torch.exp(-ce_loss)
        #     loss_dict[names[i-1]] = self.alpha * ((1-pt)**self.gamma) * ce_loss      

        # 计算总损失
        total_loss /= 3  # 减去背景类
        loss_dict['total_loss'] = total_loss

        return loss_dict

            
"""
ELoss  # TODO: 待实验

"""

class WDiceLoss():
    
    def __init__(self, smooth=1e-5):
        """
        smooth: 平滑值
        """
        self.smooth = smooth
        self.class_names = [
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Organic matter':0,
            'Organic pores':1,
            'Inorganic pores':2
        }
        self.num_classes = len(self.class_names)

    def __call__(self, logits, targets, weights=[0.3,0.3,0.4]):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        num_classes = logits.shape[1]
        tensor_one = torch.tensor(1)
        # logits argmax 
        preds = torch.softmax(logits, dim=1)
        list_2 = []
        # 计算权重
        for i in range(1,num_classes):           
            # single_weight
            pred = preds[:, i, ...]
            x = pred * (targets==i)
            x = x.mean()
            list_2.append(x)
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()     
        # 计算每个类别的损失
        loss_dict = {}
        total_loss = 0
        names = ['Organic matter', 'Organic pores', 'Inorganic pores'] 
        for i in range(1,num_classes):
            pred = preds[:, i, ...]
            target = targets[:, i, ...]
            # 计算总的损失
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2 * intersection) / (union + self.smooth)
            loss = tensor_one - dice
            # 加权
            zz = list_2[i-1]*(list_2[i-1] - 1) # 正则项
            loss = loss*(1 - (weights[i-1]**(1 - list_2[i-1]) + zz))
            loss_dict[names[i-1]] = loss 
            total_loss += loss 

        loss_dict['total_loss'] = total_loss

        return loss_dict

"""
DWDLoss 动态加权loss
""" 
class DWDLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        self.class_names = [
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'Organic matter':0,
            'Organic pores':1,
            'Inorganic pores':2
        }
        self.smooth = smooth
    def calculate_cnum(self, targets):
        cnum = []
        for i in range(1,4):
            cnum.append(torch.sum(targets==i))
        return torch.tensor(cnum)
    def calculate_weights(self, n, c):
        max_n = torch.max(n)
        return torch.log(max_n / n[c-1]) + 1
    def __call__(self, logits, targets):
        num_classes = logits.shape[1]
        # 处理logits
        preds = torch.softmax(logits, dim=1)
        targets = targets.to(torch.int64)
        masks = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # 类别权重
        total_loss = 0.0
        loss_dict = {'Organic matter' : 0.0, 'Organic pores' : 0.0, 'Inorganic pores' : 0.0}
        names = ['Organic matter', 'Organic pores', 'Inorganic pores'] 
        one = torch.tensor(1).to(targets.device)

        # 动态加权loss
        for i in range(1,num_classes):
            # 计算权重
            cnum = self.calculate_cnum(targets).to(targets.device)
            weight = self.calculate_weights(cnum, i).to(targets.device)

            # 计算dice + loss
            pred = preds[:, i, ...]    # 预测
            mask = masks[:, i, ...]    # 标签
            intersection = (pred * mask).sum()
            union = (pred.sum() + mask.sum())
            ip = intersection / masks[:, i, ...].sum()
            dice = (2 * intersection) / (union + self.smooth)
            # single_loss
            w_dice = weight ** (1-ip) * dice 
            loss = one - w_dice
            loss_dict[names[i-1]] = loss
            # add
            total_loss += loss
     
        # 计算总的损失
        loss_dict['total_loss'] = total_loss
        return loss_dict  

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
