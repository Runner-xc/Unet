"""
评价指标

1.Recall 召回率=TP/(TP+FN)
2.Precision 精准率=TP/(TP+FP)
3.F1_score=2/(1/R+1/P)  # 召回率和准确率的调和平均数
4.Contour Loss
5.Boundary Loss

"""
import torch
import torch.nn.functional as F


class Recall():
    # 设置转换为正样本的阈值为 0.5
    def __init__(self, true_label, predict_label, threshold=0.5):
        self.true_label = true_label
        self.predict_label = predict_label
        self.threshold = threshold

    def __call__(self):
        # 将预测值转换为二进制数
        predict_label_ = (self.predict_label > self.threshold).flaot()

        # 计算TP、FP、FN
        TP = torch.sum((predict_label_==1)&(self.true_label==1))
        FN = torch.sum((predict_label_==0)&(self.true_label==1))

        try: TP + FN == 0
            
        except ZeroDivisionError:
            print("ZeroDivisionError:TP + FN == 0")
            recall = 0.0

        recall = TP / (TP + FN)
        return recall

class Precision(recall):                # 继承recall类
    def __init__(self):
        super(precision, self).__init__(self)

    def __call__(self):
        # 将预测值转换为二进制数
        predict_label_ = (self.predict_label > self.threshold).flaot()

        # 计算TP、FP、FN
        TP = torch.sum((predict_label_==1)&(self.true_label==1))
        FP = torch.sum((predict_label_==1)&(self.true_label==0))
        precision = TP / (TP + FP)

        return precision
    
class F1_score(recall, precision):                
    def __init__(self):
        super(F1_score, self).__init__(self)

    def __call__(self):
        F1_score = 2 / (1/self.recall + 1/self.precision)
        return F1_score

class ContourLoss(nn.Module):
    """
    ContourLoss 类用于计算预测分割和真实分割之间的轮廓损失。
    
    参数:
        delta (float): 一个小常数，用于确保数值稳定性。
        reduction (str): 批处理预测损失的缩减方法。
    """
    
    def __init__(self, delta=1.0, reduction='mean'):
        super(ContourLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算 Contour Loss。
        
        参数:
            pred (Tensor): 预测的分割图，形状为 (N, 1, H, W)。
            target (Tensor): 真实的分割图，形状为 (N, 1, H, W) 或 (N, H, W)。
        
        返回:
            Tensor: Contour 损失值。
        """
        if target.dim() == 2:
            target = target.unsqueeze(1)  # 确保 target 与 pred 维度相同
        
        # 应用边缘检测算子（例如，Sobel 算子）以突出轮廓
        pred_edges = F.sobel(torch.sigmoid(pred))  # 假设 pred 是原始 logits
        target_edges = F.sobel(target)
        
        # 计算轮廓之间的距离
        contour_distance = (pred_edges - target_edges) ** 2
        
        # 可以应用某种形式的池化操作将距离图简化为单一值
        contour_distance = torch.mean(contour_distance)  # 示例：距离的平均值
        
        # 应用缩减方法
        if self.reduction == 'mean':
            loss = contour_distance.mean()
        elif self.reduction == 'sum':
            loss = contour_distance.sum()
        else:
            raise ValueError(f"无效的缩减方法 '{self.reduction}'")

        return loss

import torch
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    """
    Boundary Loss 类用于计算预测分割与真实分割之间的边界损失。
    
    参数:
        delta (float): 用于控制边界距离的缩放因子。
        reduction (str): 指定损失的缩减方式，'mean' 或 'sum'。
    """
    
    def __init__(self, delta=1.0, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算边界损失。
        
        参数:
            pred (Tensor): 预测的分割图，形状为 (N, C, H, W)。
            target (Tensor): 真实的分割图，形状为 (N, H, W)。
        
        返回:
            Tensor: 边界损失值。
        """
        # 确保维度匹配，如果target不是单通道的，则unsqueeze
        if target.dim() != pred.dim():
            target = target.unsqueeze(1)
        
        # 应用Sobel算子来获取边界特征
        pred_edges = self.sobel(pred)
        target_edges = self.sobel(target)
        
        # 计算边界距离
        distance = torch.abs(pred_edges - target_edges)
        
        # 根据设定的缩减方式来缩减损失
        if self.reduction == 'mean':
            loss = torch.mean(distance)
        elif self.reduction == 'sum':
            loss = torch.sum(distance)
        else:
            raise ValueError(f"Invalid reduction mode '{self.reduction}'")
        
        return loss

    def sobel(self, img):
        """
        使用 Sobel 算子计算图像的梯度。
        
        参数:
            img (Tensor): 输入的图像张量。
        
        返回:
            Tensor: Sobel 算子计算的梯度。
        """
        # 定义 Sobel 卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device)
        
        # 应用 Sobel 卷积核
        grad_x = F.conv2d(img, sobel_x.unsqueeze(0).unsqueeze(0), padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, sobel_y.unsqueeze(0).unsqueeze(0), padding=1, groups=img.shape[1])
        
        # 计算梯度幅度
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return grad_magnitude
