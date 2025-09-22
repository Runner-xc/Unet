from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch
import numpy as np
import torch.nn.functional as F

"""
训练和验证
"""

def _total_loss(model_outputs, target, loss_fn, class_names):
    """
    一个辅助函数，用于处理像U2-Net这样有多个输出的模型。
    """
    # 创建损失字典
    loss_dict = {cls: 0.0 for cls in class_names}
    loss_dict['total_loss'] = 0.0
    
    # 获取每一层的损失
    # 假设 loss_fn 返回一个字典 {'class1': loss1, 'total_loss': total}
    loss_dict_list = [loss_fn(F.softmax(model_output, dim=1), target) for model_output in model_outputs]

    for i in loss_dict_list: 
        for cls in class_names:
            loss_dict[cls] += i[cls]
        loss_dict['total_loss'] += i['total_loss'] 
    
    # 计算平均损失
    for cls in class_names:
        loss_dict[cls] /= len(loss_dict_list)
    loss_dict['total_loss'] /= len(loss_dict_list)
    return loss_dict

def train_one_epoch_cfg(model, optimizer, epoch, train_dataloader, device, loss_fn, scaler, Metric, scheduler, cfg):
    """
    在一个epoch上训练模型。
    
    :param model: 模型
    :param optimizer: 优化器
    :param epoch: 当前epoch
    :param train_dataloader: 训练数据集
    :param device: 设备
    :param loss_fn: 损失函数
    :param scaler: 梯度缩放器 (用于混合精度)
    :param Metric: 指标计算器
    :param scheduler: 学习率调度器
    :param cfg: yacs配置节点对象 (所有配置的来源)
    """
    model.train()
    
    # 从配置中获取必要的参数
    class_names = cfg.DATA.CLASS_NAMES
    use_eln_loss = cfg.TRAIN.LOSS.ELN_LOSS
    l1_lambda = cfg.TRAIN.LOSS.L1_LAMBDA
    l2_lambda = cfg.TRAIN.LOSS.L2_LAMBDA
    
    Metric_list = np.zeros((6, len(class_names) + 1))
    train_dataloader_tqdm = tqdm(train_dataloader, desc=f" Training on Epoch :{epoch + 1}😀", leave=False)
    epoch_losses = [0.0] * (len(class_names) + 1)  # 加上总损失
    
    for data in train_dataloader_tqdm: 
        images, masks = data[0][0], data[0][1]
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type="cuda", enabled=cfg.TRAIN.AMP):
            pred = model(images)  
            masks = masks.to(torch.int64).squeeze(1) # 确保masks是正确形状

            total_loss = 0
            
            # --- 统一的损失计算逻辑 ---
            if isinstance(pred, list): # U2Net case
                loss_dict = _total_loss(pred, masks, loss_fn, class_names)
                losses = list(loss_dict.values())
                total_loss = loss_dict['total_loss']
                # 使用第一个输出来计算指标
                metrics = Metric.update(pred[0], masks)
                Metric_list += metrics

            elif isinstance(pred, tuple) and len(pred) == 2: # 辅助分类器 case
                heatmap, aux = pred
                main_loss_dict = loss_fn(heatmap, masks)
                aux_loss_dict = loss_fn(aux, masks)
                
                losses = [m_loss * 0.6 + a_loss * 0.4 for m_loss, a_loss in zip(main_loss_dict.values(), aux_loss_dict.values())]
                total_loss = losses[-1]
                metrics = Metric.update(heatmap, masks)
                Metric_list += metrics
                
            else: # 标准单输出 case
                loss_dict = loss_fn(pred, masks)
                losses = list(loss_dict.values())
                total_loss = loss_dict['total_loss']
                metrics = Metric.update(pred, masks)
                Metric_list += metrics

            # --- 统一的正则化逻辑 ---
            if use_eln_loss:
                elastic_net_loss = model.elastic_net(l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                total_loss = total_loss + elastic_net_loss
        
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]

    Metric_list /= len(train_dataloader)
    return epoch_losses, Metric_list   

def evaluate_cfg(model, device, data_loader, loss_fn, Metric, cfg):
    """
    在验证集或测试集上评估模型。

    :param model: 模型
    :param device: 设备
    :param data_loader: 数据集
    :param loss_fn: 损失函数
    :param Metric: 指标计算器
    :param cfg: yacs配置节点对象
    """
    model.eval()
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    class_names = cfg.DATA.CLASS_NAMES
    Metric_list = np.zeros((6, len(class_names) + 1))
    epoch_losses = [0.0] * (len(class_names) + 1)

    with torch.no_grad():
        val_dataloader_tqdm = tqdm(data_loader, desc="  Validating  😀", leave=False)
        for data in val_dataloader_tqdm:
            images, masks = data[0][0].to(device), data[0][1].to(device)
            with autocast(device_type="cuda", enabled=cfg.TRAIN.AMP):
                pred_mask = model(images)
                masks = masks.to(torch.int64).squeeze(1)

                # --- 统一的损失和指标计算逻辑 ---
                if isinstance(pred_mask, list):
                    loss_dict = _total_loss(pred_mask, masks, loss_fn, class_names)
                    losses = list(loss_dict.values())
                    metrics = Metric.update(pred_mask[0], masks)

                elif isinstance(pred_mask, tuple) and len(pred_mask) == 2:
                    heatmap, aux = pred_mask
                    main_loss_dict = loss_fn(heatmap, masks)
                    aux_loss_dict = loss_fn(aux, masks)
                    losses = [m*0.6 + a*0.4 for m, a in zip(main_loss_dict.values(), aux_loss_dict.values())]
                    metrics = Metric.update(heatmap, masks)
                
                else:
                    loss_dict = loss_fn(pred_mask, masks)
                    losses = list(loss_dict.values())
                    metrics = Metric.update(pred_mask, masks)
                
                Metric_list += metrics
                epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]

    Metric_list /= len(data_loader)
    return epoch_losses, Metric_list