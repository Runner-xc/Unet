"""
评价指标

1.Recall 召回率=TP/(TP+FN)
2.Precision 精准率=TP/(TP+FP)
3.F1_score=2/(1/R+1/P)  # 召回率和准确率的调和平均数
4.Contour Loss
5.Boundary Loss

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Evaluate_Metric(nn.Module):
    def __init__(self, smooth=1e-5):
        super(Evaluate_Metric, self).__init__()
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
        self.smooth = smooth
        
    def compute_confusion_matrix(self, img_pred, img_mask, threshold=0.5):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, 1, h, w) -> one_hot (batch, 4, h, w)
        """
        
        # 将预测概率转换为二进制值
        img_pred_binary = (img_pred > threshold).to(torch.int64)
        img_mask = img_mask.to(torch.int64)

        # 计算混淆矩阵的元素
        TP = ((img_pred_binary & img_mask) == 1).sum(dim=(-2, -1))
        FN = ((~img_pred_binary & img_mask) == 1).sum(dim=(-2, -1))
        FP = ((img_pred_binary & ~img_mask) == 1).sum(dim=(-2, -1))
        TN = ((~img_pred_binary & ~img_mask) == 1).sum(dim=(-2, -1))

        return TP, FN, FP, TN

    def recall(self, img_pred, img_mask):
        """"
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        # recall_dict = {}
        # class_names = self.class_names

        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=4).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=4).permute(0, 3, 1, 2).float() 

        # 计算总体召回率
        TP, FN, _, _ = self.compute_confusion_matrix(img_pred, img_mask)
        recall = TP / (TP + FN + self.smooth)
        recall = recall.mean(dim=0)
        
        OM_rc = recall[1].item()
        OP_rc = recall[2].item()
        IOP_rc = recall[3].item()
        recall = recall.sum() / 4
        recall = recall.item()
        
        return OM_rc, OP_rc, IOP_rc, recall

    
    def precision(self, img_pred, img_mask, threshold=0.5):
        # precision_dict = {}
        # class_names = self.class_names

        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=4).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=4).permute(0, 3, 1, 2).float() 
        
        # 计算总体精准率
        TP, _, FP, _ = self.compute_confusion_matrix(img_pred, img_mask)
        precision = TP / (TP + FP + self.smooth)
        precision = precision.mean(dim=0)
        
        OM_pc = precision[1].item()
        OP_pc = precision[2].item()
        IOP_pc = precision[3].item()
        precision = precision.sum() / 4
        precision = precision.item()
        
        return OM_pc, OP_pc, IOP_pc, precision


    def f1_score(self, img_pred, img_mask): 
        """
        recall:     召回率
        precision:  精准率
        """               
        OM_rc, OP_rc, IOP_rc, recall = self.recall(img_pred, img_mask)
        OM_pc, OP_pc, IOP_pc, precision = self.precision(img_pred, img_mask)

        # OM_F1
        OM_F1 = 2 * (OM_rc * OM_pc) / (OM_rc + OM_pc + self.smooth)

        # OP_F1
        OP_F1 = 2 * (OP_rc * OP_pc) / (OP_rc + OP_pc + self.smooth)

        # IOP_F1
        IOP_F1 = 2 * (IOP_rc * IOP_pc) / (IOP_rc + IOP_pc + self.smooth)

        # F1_score
        F1_score = 2 * (recall * precision) / (recall + precision + self.smooth)
        
        return OM_F1, OP_F1, IOP_F1, F1_score
    

    def dice_coefficient(self, logits, targets):
        """
        dice 指数
        """
        num_classes = logits.shape[1]
        # 预处理
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # targets: (b, h, w) -> (b, c, h, w)
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float() 
        
        # 计算总体dice
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1))
        dice = (2 * intersection) / (union + self.smooth)
        
        OM_dice = dice[1].item()
        OP_dice = dice[2].item()
        IOP_dice = dice[3].item()
        dice = dice.mean()
        dice = dice.item()

        return OM_dice, OP_dice, IOP_dice, dice

    def mIoU(self, logits, targets):
        """
        mIoU: 平均交并比
        """
        num_classes = logits.shape[1]
        logits = torch.argmax(logits, dim=1)
        logits = F.one_hot(logits, num_classes=num_classes).permute(0, 3, 1, 2).float()
        targets = targets.to(torch.int64)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 计算总体mIoU
        intersection = (logits * targets).sum(dim=(0,-2,-1))
        union = logits.sum(dim=(0,-2,-1)) + targets.sum(dim=(0,-2,-1)) - intersection
        iou =  intersection / (union + self.smooth)
        
        OM_iou = iou[1].item()
        OP_iou = iou[2].item()
        IOP_iou = iou[3].item()
        mIoU = iou.mean()
        mIoU = mIoU.item()
        
        return OM_iou, OP_iou, IOP_iou, mIoU
    
    def accuracy(self, logits, targets):
        """
        accuracy: 准确率
        """
        # 预处理
        img_pred = torch.argmax(logits, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=4).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(targets, num_classes=4).permute(0, 3, 1, 2).float() 
        
        # 计算总体精准率
        TP, FN, FP, TN = self.compute_confusion_matrix(img_pred, img_mask)
        
        accuracy = (TP + TN) / (TP + TN + FN + FP + self.smooth)
        accuracy = accuracy.mean(dim=0)
        OM_acc = accuracy[1].item()
        OP_acc = accuracy[2].item()
        IOP_acc = accuracy[3].item()
        accuracy = accuracy.mean().item()
        
        return OM_acc, OP_acc, IOP_acc, accuracy

    def update(self, img_pred, img_mask):
        """
        更新评价指标
        """
        recall = self.recall(img_pred, img_mask)
        precision = self.precision(img_pred, img_mask)
        dice = self.dice_coefficient(img_pred, img_mask)
        f1_score = self.f1_score(img_pred, img_mask)
        mIoU = self.mIoU(img_pred, img_mask)
        accuracy = self.accuracy(img_pred, img_mask)
        
        metrics = [recall, precision, dice, f1_score, mIoU, accuracy]
        metrics = np.stack(metrics, axis=0)
        metrics = np.nan_to_num(metrics)

        return metrics