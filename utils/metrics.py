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
        self.class_names = ['background',
                            'Organic matter', 
                            'Organic pores', 
                            'Inorganic pores']
        self.labels = {
            'background':0,
            'Organic matter':1,
            'Organic pores':2,
            'Inorganic pores':3
        }
        self.num_classes = len(self.labels)

    def split_class(self, img_pred, img_mask):
        """
        img_pred: 预测值 (batch, 3, h, w)
        img_mask: 标签值 (batch, 1, h, w) -> one_hot (batch, 3, h, w)
        """

        class_0 = img_pred[:, 0, ...]
        class_1 = img_pred[:, 1, ...]
        class_2 = img_pred[:, 2, ...]
        class_3 = img_pred[:, 3, ...]
        pre_class_list = [class_1, class_2, class_3]  # 舍弃背景类

        img_mask = F.one_hot(img_mask, self.num_classes).float()
        mask_0 = img_mask[:, 0, ...]
        mask_1 = img_mask[:, 1, ...]
        mask_2 = img_mask[:, 2, ...]
        mask_3 = img_mask[:, 3, ...]
        mask_class_list = [mask_1, mask_2, mask_3]

        return pre_class_list, mask_class_list

    def Recall(self, img_pred, img_mask, threshold=0.5):
        """"
        img_pred: 预测值 (batch, 4, h, w)
        img_mask: 标签值 (batch, h, w)
        """
        recall_dict = {}
        class_names = self.class_names
        # 获取类别的预测值和标签
        pred_class_list, mask_class_list = self.split_class(img_pred, img_mask)
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            # 将预测值转换为二进制数
            predict_label = (pred_class > threshold).float()
            # 计算TP、FP、FN
            TP = torch.sum((predict_label==1)&(mask_class==1))
            FN = torch.sum((predict_label==0)&(mask_class==1))
            # 计算召回率
            try: TP + FN == 0  
            except ZeroDivisionError:
                print("ZeroDivisionError:TP + FN == 0")
                recall = 0.0
            recall = TP / (TP + FN)
            recall_dict[class_name] = recall
        
        OM_rc = recall_dict['Organic matter']
        OP_rc = recall_dict['Organic pores']
        IOP_rc = recall_dict['Inorganic pores']
        recall = (OM_rc + OP_rc + IOP_rc) / len(class_names)

        return OM_rc, OP_rc, IOP_rc, recall

    def Precision(self, img_pred, img_mask, threshold=0.5):
        precision_dict = {}
        class_names = self.class_names
        # 获取类别的预测值和标签
        pred_class_list, mask_class_list = self.split_class(img_pred, img_mask)
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            # 将预测值转换为二进制数
            predict_label = (pred_class > threshold).float()
            # 计算TP、FP、FN
            TP = torch.sum((predict_label==1)&(mask_class==1))
            FP = torch.sum((predict_label==1)&(mask_class==0))
            # 计算精准率
            try: TP + FP == 0  
            except ZeroDivisionError:
                print("ZeroDivisionError:TP + FP == 0")
                precision = 0.0
            precision = TP / (TP + FP)
            precision_dict[class_name] = precision

        OM_pc = precision_dict['Organic matter']
        OP_pc = precision_dict['Organic pores']
        IOP_pc = precision_dict['Inorganic pores']
        precision = (OM_pc + OP_pc + IOP_pc) / len(class_names)

        return OM_pc, OP_pc, IOP_pc, precision

    def F1_score(self, img_pred, img_mask): 
        """
        recall:     召回率
        precision:  精准率
        """               
        OM_rc, OP_rc, IOP_rc, recall = self.Recall(img_pred, img_mask)
        OM_pc, OP_pc, IOP_pc, precision = self.Precision(img_pred, img_mask)

        # OM_F1
        OM_F1 = 2 * (OM_rc * OM_pc) / (OM_rc + OM_pc)

        # OP_F1
        OP_F1 = 2 * (OP_rc * OP_pc) / (OP_rc + OP_pc)

        # IOP_F1
        IOP_F1 = 2 * (IOP_rc * IOP_pc) / (IOP_rc + IOP_pc)

        # F1_score
        F1_score = 2 * (recall * precision) / (recall + precision)
        
        return OM_F1, OP_F1, IOP_F1, F1_score

    def Dice(self, img_pred, img_mask, smooth=1e-5):
        """
        dice 指数
        """
        smooth = smooth
        dice_dict = {}
        class_names = self.class_names
        img_pred = torch.argmax(img_pred, dim=1)
        img_pred = F.one_hot(img_pred, self.num_classes).permute(0, 3, 1, 2).float()
        pred_class_list, mask_class_list = self.split_class(img_pred, img_mask)
        # 计算每个类的dice
        for pred_class, mask_class, class_name in zip(pred_class_list, mask_class_list, class_names):
            intersection = (pred_class * mask_class).sum()
            union = pred_class.sum() + mask_class.sum()
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_dict[class_name] = dice
        
        OM_dice = dice_dict['Organic matter']
        OP_dice = dice_dict['Organic pores']
        IOP_dice = dice_dict['Inorganic pores']
        dice = (OM_dice + OP_dice + IOP_dice) / len(class_names)

        return OM_dice, OP_dice, IOP_dice, dice

    def update(self, img_pred, img_mask):
        """
        更新评价指标
        """
        recall = self.Recall(img_pred, img_mask)
        precision = self.Precision(img_pred, img_mask)
        dice = self.Dice(img_pred, img_mask)
        f1_score = self.F1_score(img_pred, img_mask)

         # 确保每个指标都是一个数组，并将它们移动到 CPU
        recall_np = recall.cpu().numpy() if isinstance(recall, torch.Tensor) else np.array(recall)
        precision_np = precision.cpu().numpy() if isinstance(precision, torch.Tensor) else np.array(precision)
        dice_np = dice.cpu().numpy() if isinstance(dice, torch.Tensor) else np.array(dice)
        f1_score_np = f1_score.cpu().numpy() if isinstance(f1_score, torch.Tensor) else np.array(f1_score)

        # 创建一个包含所有指标的列表
        metrics = [recall_np, precision_np, dice_np, f1_score_np]

        # 堆叠指标
        metrics_np = np.stack(metrics, axis=0)

        return metrics_np