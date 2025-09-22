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
    model_output: 预测值
    target: 真实值
    loss_fn: 损失函数
    """
    # 创建损失字典
    loss_dict = {cls:0.0 for cls in class_names}
    loss_dict['total_loss'] = 0.0
    # 获取每一层的损失
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

def train_one_epoch(model, optimizer, epoch, train_dataloader, device, loss_fn, scaler, Metric, scheduler, class_names, elnloss, l1_lambda, l2_lambda):
    """"
    model:             模型
    optimizer:         优化器
    epoch:             当前epoch
    train_dataloader:  训练数据集
    device:            设备
    loss_fn:           损失函数
    scaler:            梯度缩放器
    scheduler:         调度器
    elnloss:           是否使用Elastic Net正则化
    l1_lambda:         l1正则化系数
    l2_lambda:         l2正则化系数
    """
    model.train()
    Metric_list = np.zeros((6, len(class_names)+1))
    train_dataloader = tqdm(train_dataloader, desc=f" Training on Epoch :{epoch + 1}😀", leave=False)
    epoch_losses = [0.0] * len(class_names+ ['total'])  # 加上总损失
    
    for data in train_dataloader: 
        # 获取训练数据集的一个batch
        images, masks = data[0][0], data[0][1]
        images, masks = images.to(device), masks.to(device)
        # 梯度清零
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(device_type="cuda"):
            pred = model(images)  
            masks = masks.to(torch.int64)

            # U2Net
            if isinstance(pred, list):
                loss_dict = _total_loss(pred, masks, loss_fn, class_names)  #  训练输出 7 个预测结果，6 个解码器输出和 1 个总输出。
                losses = loss_dict.values()
                total_loss = loss_dict['total_loss']
                if elnloss:
                    # 添加Elastic Net正则化
                    elastic_net_loss = model.elastic_net(l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                    total_loss = total_loss + elastic_net_loss
                metrics = Metric.update(pred, masks)
                Metric_list += metrics
            
            # 是否使用辅助分类器
            elif isinstance(pred, tuple):
                if len(pred) == 2:
                    heatmap, aux = pred
                    # 主分支loss
                    main_loss_dict = loss_fn(heatmap, masks)
                    main_losses = main_loss_dict.values()
                    
                    # 辅助分支loss
                    aux_loss_dict = loss_fn(aux, masks)
                    aux_losses = aux_loss_dict.values()
                    
                    # 计算总损失：主分支损失*0.6 + 辅助分支损失*0.4
                    losses = [m_loss*0.6 + a_loss*0.4 for m_loss, a_loss in zip(main_losses, aux_losses)]
                    total_loss = losses[-1]
                    if elnloss:
                        # 添加Elastic Net正则化
                        elastic_net_loss = model.elastic_net(l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                        total_loss = total_loss + elastic_net_loss
                    metrics = Metric.update(heatmap, masks)
                    Metric_list += metrics
            else:
                loss_dict = loss_fn(pred, masks)
                losses = loss_dict.values()
                total_loss = loss_dict['total_loss']
                if elnloss:
                    # 添加Elastic Net正则化
                    elastic_net_loss = model.elastic_net(l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                    total_loss = total_loss + elastic_net_loss
                metrics = Metric.update(pred, masks)
                Metric_list += metrics

        # 反向传播
        scaler.scale(total_loss).backward()
      
        # 检查梯度是否包含inf或nan
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_losses = [x+y for x, y in zip(epoch_losses, losses)]

    Metric_list /= len(train_dataloader)
    return  epoch_losses, Metric_list   

def evaluate(model, device, data_loader, loss_fn, Metric, class_names, test:bool=False):
    """
    model:       模型
    device:      设备
    data_loader: 数据集
    loss_fn:     损失函数
    Metric:      指标
    """
    model.eval()
    if test:
        Metric_list = np.zeros((6, len(class_names)+1))
    else:
        Metric_list = np.zeros((6, len(class_names)+1))
    epoch_losses = [0.0] * len(class_names+['total'])  # 加上总损失

    with torch.no_grad():
        val_dataloader = tqdm(data_loader, desc=f"  Validating  😀", leave=False)
        for data in val_dataloader:
            images, masks =data[0][0].to(device), data[0][1].to(device)
            with autocast(device_type="cuda"):
                pred_mask = model(images)         # 验证  模型 softmax 输出
                masks = masks.to(torch.int64)
                masks = masks.squeeze(1)
                # U2Net
                if isinstance(pred_mask, list):
                    loss_dict = _total_loss(pred_mask, masks, loss_fn, class_names)  #  训练输出 7 个预测结果，6 个解码器输出和 1 个总输出。
                    losses = loss_dict.values()
                    total_loss = loss_dict['total_loss']
                    metrics = Metric.update(pred_mask, masks)
                    Metric_list += metrics

                # 是否使用辅助分类器
                elif isinstance(pred_mask, tuple):
                    heatmap, aux = pred_mask
                    # 主分支loss
                    main_loss_dict = loss_fn(heatmap, masks)
                    main_losses = main_loss_dict.values()
                    # 辅助分支loss
                    aux_loss_dict = loss_fn(aux, masks)
                    aux_losses = aux_loss_dict.values()
                    # 计算总损失：主分支损失*0.6 + 辅助分支损失*0.4
                    losses = [m_loss*0.6 + a_loss*0.4 for m_loss, a_loss in zip(main_losses, aux_losses)]
                    total_loss = losses[-1]
                    metrics = Metric.update(heatmap, masks)
                    Metric_list += metrics    

                else:
                    loss_dict = loss_fn(pred_mask, masks)
                    losses = loss_dict.values()
                    metrics = Metric.update(pred_mask, masks)
                    Metric_list += metrics    

            # 累加损失 
            epoch_losses = [x+y for x, y in zip(epoch_losses, losses)]
    Metric_list /= len(val_dataloader)
    return  epoch_losses, Metric_list
