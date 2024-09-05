"""
损失函数:Dice Loss、BoundaryLoss、Focal Loss、DistanceBasedLoss、Tversky Loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

 
class DiceLoss(torch.nn.Module):
    
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, pred, target):
        pred = torch.sigmoid(pred) # 首先对预测结果进行sigmoid转换
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss
    

class BoundaryLoss(nn.Module):
    """
    边界损失函数,使用距离正则化项来鼓励模型在物体边缘处的精确预测。
    
    参数:
        delta (float): 边界宽度,用于确定边界区域。
        gamma (float): 用于控制距离正则化项的权重。
    """
    def __init__(self, delta=1.0, gamma=1.0):
        super(BoundaryLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, target):
        """
        计算边界损失。
        
        参数:
            pred (Tensor): 模型输出的预测分割图,期望形状为 (N, 1, H, W)。
            target (Tensor): 真实分割图,形状为 (N, 1, H, W) 或 (N, H, W)。
        
        返回:
            Tensor: 边界损失值。
        """
        if target.dim() == 2:
            target = target.unsqueeze(1)  # 确保target是二维的

        # 计算边界区域
        target_edges = self._compute_edges(target)

        # 计算预测分割和真实分割之间的差异
        diff = torch.abs(pred - target)

        # 计算边界损失
        boundary_diff = diff * target_edges
        loss = (boundary_diff.mean() - self.delta) ** 2

        # 计算距离正则化项
        dist_reg = self.gamma * (torch.exp(-boundary_diff) - 1).mean()

        # 总损失是边界损失和距离正则化项的和
        total_loss = loss + dist_reg

        return total_loss

    def _compute_edges(self, target):
        """
        使用Sobel算子计算目标分割图的边缘。
        
        参数:
            target (Tensor): 真实分割图。
        
        返回:
            Tensor: 边缘响应图。
        """
        # 定义Sobel滤波器
        sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        sobel_kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0)  # 扩展维度以匹配输入
        sobel_kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0)

        # 使用Sobel算子计算水平和垂直边缘
        target_x = F.conv2d(target, sobel_kernel_x, padding=1, groups=1)
        target_y = F.conv2d(target, sobel_kernel_y, padding=1, groups=1)

        # 计算边缘响应
        target_edges = torch.sqrt(target_x ** 2 + target_y ** 2)

        return target_edges
    

class Focal_Loss():
    """
    γ : 聚焦因子,用于控制损失的敏感度
    weight : 平衡正负样本权重
    """
    def __init__(self,weight,gamma=2):
        super(Focal_Loss,self).__init__()
        self.gamma=gamma
        self.weight=weight
    def forward(self,preds,labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps=1e-7
        y_pred =preds.view((preds.size()[0],preds.size()[1],-1)) #B*C*H*W->B*C*(H*W)
        
        target=labels.view(y_pred.size()) #B*C*H*W->B*C*(H*W)
        
        ce=-1*torch.log(y_pred+eps)*target
        floss=torch.pow((1-y_pred),self.gamma)*ce
        floss=torch.mul(floss,self.weight)
        floss=torch.sum(floss,dim=1)
        return torch.mean(floss)
        

class DistanceBasedLoss(nn.Module):
    """
    基于距离的损失函数,计算预测分割与真实分割之间的平均表面距离。

    参数:
        reduction (str): 指定损失的缩减方式,'mean' 或 'sum'。
    """
    def __init__(self, reduction='mean'):
        super(DistanceBasedLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算基于距离的损失。

        参数:
            pred (Tensor): 模型输出的预测分割图,期望形状为 (N, 1, H, W)。
            target (Tensor): 真实分割图,形状为 (N, 1, H, W) 或 (N, H, W)。

        返回:
            Tensor: 基于距离的损失值。
        """
        # 确保维度一致
        if target.dim() == 2:
            target = target.unsqueeze(1)
        
        # 计算预测分割与真实分割之间的差异
        diff = torch.abs(pred - target)

        # 计算边界像素的索引
        border = target ^ (target > 0).all(dim=1, keepdim=True)  # 找到边界
        diff_border = diff * border.float()

        # 计算表面距离
        distances = F.avg_pool2d(diff_border, kernel_size=3, stride=1, padding=1)

        # 根据缩减方式处理损失
        if self.reduction == 'mean':
            loss = distances.mean()
        elif self.reduction == 'sum':
            loss = distances.sum()
        else:
            raise ValueError("Invalid reduction mode: '{}'".format(self.reduction))

        return loss



class TverskyLoss(nn.Module):
    """
    Tversky Loss 类用于图像分割任务，可以在精度和召回率之间取得平衡。
    
    参数:
        alpha (float): 假正例的权重。
        beta (float): 假负例的权重。
        smooth (float): 一个小常数，用于确保数值稳定性。
        reduction (str): 批处理预测的损失减少方法。
    """
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算 Tversky Loss。
        
        参数:
            pred (Tensor): 每个类别的预测概率，形状为 (N, C, H, W)。
            target (Tensor): 真实标签，形状为 (N, H, W)。
        
        返回:
            Tensor: Tversky Loss 的值。
        """
        # 确保预测结果在概率尺度上，并且与目标形状相同
        pred = torch.sigmoid(pred)
        
        # 计算真正例、假正例和假负例
        true_positives = (pred * target).sum(dim=(1, 2, 3))
        false_positives = (pred * (1 - target)).sum(dim=(1, 2, 3))
        false_negatives = ((1 - pred) * target).sum(dim=(1, 2, 3))
        
        # 计算 Tversky 指数
        tversky_index = (true_positives + self.smooth) / (true_positives + 
                          self.alpha * false_positives + 
                          self.beta * false_negatives + self.smooth)
        
        # 计算 Tversky Loss
        loss = 1 - tversky_index
        
        # 应用缩减方法
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            # 不进行缩减，返回每个元素的损失
            pass
        else:
            raise ValueError(f"无效的缩减方法 '{self.reduction}'。")
        
        return loss

class TotalLoss(nn.Module):
    """"
    loss_fn : 'tversky'、'dice'、'boundary'、'focal'、'distancebased' 
    """
    def __init__(self, flag: bool = True, loss_fn: str = None):
        super().__init__()
        self.flag = flag
        self.tversky_loss =TverskyLoss(alpha=0.5, beta=0.5, smooth=1e-6, reduction='mean')
        self.dice_loss = Dice_Loss()
        self.boundary_loss = BoundaryLoss()
        self.focal_loss = FocalLoss()
        self.distancebased_Loss = DistanceBasedLoss()
        self.loss_fn = loss_fn

    def __call__(self, pred, labels):
        # 判断是否使用多个损失函数组合
        if self.flag:
            return self.tversky_loss(pred, labels) + self.dice_loss(pred, labels) + self.boundary_loss(pred, labels) + self.focal_loss(pred, labels) + self.distancebased_Loss(pred, labels)
        else:
            # 'tversky'
            if self.loss_fn == 'tversky':
                return self.tversky_loss(pred, labels)
            # 'dice'
            elif self.loss_fn == 'dice':
                return self.dice_loss(pred, labels)
            # 'boundary'
            elif self.loss_fn == 'boundary':
                return self.boundary_loss(pred, labels)
            # 'focal'
            elif self.loss_fn == 'focal':
                return self.focal_loss(pred, labels)
            # 'distancebased'
            elif self.loss_fn == 'distancebased':
                return self.distancebased_Loss(pred, labels)
            
            else:
                raise ValueError(f"无效损失函数{self.loss_fn}")

if __name__ == '__main__':
    loss_fn = TotalLoss(flag=True, loss_fn='tversky')
    x = torch.rand(1, 1, 256, 256)
    y = torch.rand(1, 1, 256, 256)
    print(loss_fn(x, y))