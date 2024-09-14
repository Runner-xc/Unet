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
    def __init__(self):
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
        self.num_classes = len(self.labels)

    def split_class(self, img_pred, img_mask):
        """
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, 1, h, w) -> one_hot (batch, 4, h, w)
        """

        class_0 = img_pred[:, 0, ...]
        class_1 = img_pred[:, 1, ...]
        class_2 = img_pred[:, 2, ...]
        class_3 = img_pred[:, 3, ...]
        pre_class_list = [class_0, class_1, class_2, class_3]  # 舍弃背景类

        mask_0 = img_mask[:, 0, ...]
        mask_1 = img_mask[:, 1, ...]
        mask_2 = img_mask[:, 2, ...]
        mask_3 = img_mask[:, 3, ...]
        mask_class_list = [mask_0, mask_1, mask_2, mask_3]

        return pre_class_list, mask_class_list
    def compute_confusion_matrix(self, img_pred, img_mask):
        """
        计算 TP FP TN FN
        """
        assert img_pred.shape == img_mask.shape
        tensor_one = torch.tensor(1)

        # 计算混淆矩阵的元素
        TP = (img_pred * img_mask).sum(dim=(-2, -1)) # 预测为正类，实际也为正类
        FN = ((tensor_one - img_pred)*img_mask).sum(dim=(-2, -1)) # 预测为负类，实际为正类
        FP = (img_pred * (tensor_one - img_mask)).sum(dim=(-2, -1)) # 预测为正类，实际为负类
        TN = ((tensor_one - img_pred) * (tensor_one - img_mask)).sum(dim=(-2, -1)) # 预测为负类，实际也为负类
        return TP, FN, FP, TN

    def recall(self, img_pred, img_mask):
        """"
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        recall_dict = {}
        class_names = self.class_names

        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=4).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=4).permute(0, 3, 1, 2).float() 
    

        # 获取类别的预测值和标签
        pred_class_list, mask_class_list = self.split_class(img_pred, img_mask)

        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            TP, FN, _, _ = self.compute_confusion_matrix(pred_class, mask_class)
            recall = TP / (TP + FN)
            recall_dict[class_name] = recall
        
        OM_rc = recall_dict['Organic matter'].item()
        OP_rc = recall_dict['Organic pores'].item()
        IOP_rc = recall_dict['Inorganic pores'].item()
        recall = sum(recall_dict.values() - recall_dict['Background']).item() / (len(class_names) - 1) 

        return OM_rc, OP_rc, IOP_rc, recall

    
    def precision(self, img_pred, img_mask, threshold=0.5):
        precision_dict = {}
        class_names = self.class_names

        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=4).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=4).permute(0, 3, 1, 2).float() 

        # 获取类别的预测值和标签
        pred_class_list, mask_class_list = self.split_class(img_pred, img_mask)
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            TP, _, FP, _ = self.compute_confusion_matrix(pred_class, mask_class)
            precision = TP / (TP + FP)
            precision_dict[class_name] = precision

        OM_pc = precision_dict['Organic matter'].item()
        OP_pc = precision_dict['Organic pores'].item()
        IOP_pc = precision_dict['Inorganic pores'].item()
        precision = sum(precision_dict.values() - precision_dict['Background']).item() / (len(class_names) - 1)

        return OM_pc, OP_pc, IOP_pc, precision


    def f1_score(self, img_pred, img_mask): 
        """
        recall:     召回率
        precision:  精准率
        """               
        OM_rc, OP_rc, IOP_rc, recall = self.recall(img_pred, img_mask)
        OM_pc, OP_pc, IOP_pc, precision = self.precision(img_pred, img_mask)

        # OM_F1
        if (OM_rc + OM_pc) == 0:
            OM_F1 = 0.0
        else:
            OM_F1 = 2 * (OM_rc * OM_pc) / (OM_rc + OM_pc)

        # OP_F1
        if (OP_rc + OP_pc) == 0:
            OP_F1 = 0.0
        else:
            OP_F1 = 2 * (OP_rc * OP_pc) / (OP_rc + OP_pc)

        # IOP_F1
        if (IOP_rc + IOP_pc) == 0:
            IOP_F1 = 0.0
        else:
            IOP_F1 = 2 * (IOP_rc * IOP_pc) / (IOP_rc + IOP_pc)

        # F1_score
        if (recall + precision) == 0:
            F1_score = 0.0
        else:
            F1_score = 2 * (recall * precision) / (recall + precision)
        
        return OM_F1, OP_F1, IOP_F1, F1_score
    

    def dice_coefficient(self, img_pred, img_mask):
        """
        dice 指数
        """
        dice_dict = {}
        class_names = self.class_names
        # 预处理
        img_pred = torch.argmax(img_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        img_pred = F.one_hot(img_pred, num_classes=4).permute(0, 3, 1, 2).float() 
        img_mask = F.one_hot(img_mask, num_classes=4).permute(0, 3, 1, 2).float() 

        pred_class_list, mask_class_list = self.split_class(img_pred, img_mask)
        # 计算每个类的dice
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            intersection = (pred_class * mask_class).sum(dim=(-2,-1))
            union = pred_class.sum(dim=(-2,-1)) + mask_class.sum(dim=(-2,-1))
            dice = (2 * intersection ) / union
            dice_dict[class_name] = dice
        
        OM_dice = dice_dict['Organic matter'].item()
        OP_dice = dice_dict['Organic pores'].item()
        IOP_dice = dice_dict['Inorganic pores'].item()
        dice = sum(dice_dict.values() - dice_dict['Background']).item() / (len(class_names) - 1)

        return OM_dice, OP_dice, IOP_dice, dice

      

    def update(self, img_pred, img_mask):
        """
        更新评价指标
        """
        recall = self.recall(img_pred, img_mask)
        precision = self.precision(img_pred, img_mask)
        dice = self.dice_coefficient(img_pred, img_mask)
        f1_score = self.f1_score(img_pred, img_mask)

        metrics = [recall, precision, dice, f1_score]
        metrics = np.stack(metrics, axis=0)
        metrics = np.nan_to_num(metrics)

        return metrics