from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch
import numpy as np
import torch.nn.functional as F
"""
训练和验证
"""
def total_loss(model_output, target, loss_fn):
    """
    model_output: 预测值
    target: 真实值
    loss_fn: 损失函数
    """
    # 获取总的损失 TODO: 使用字典存储损失
    loss_dict_list = [loss_fn(F.softmax(model_output[i], dim=1), target) for i in range(len(model_output))]

    total_losses = torch.tensor(0.0, dtype=torch.float32, device="cuda:0")
    OM_losses = torch.tensor(0.0, dtype=torch.float32, device="cuda:0")
    OP_losses = torch.tensor(0.0, dtype=torch.float32, device="cuda:0")
    IOP_losses = torch.tensor(0.0, dtype=torch.float32, device="cuda:0")

    # 遍历每一层损失
    for loss_dict in loss_dict_list: 
        # OM_loss = loss_dict['Organic matter']   # list:[8]
        # OP_loss = loss_dict['Organic pores']
        # IOP_loss = loss_dict['Inorganic pores']
        total_loss = loss_dict['total_loss']
        
        # 累加损失
        total_losses += total_loss
        # OM_losses += OM_loss
        # OP_losses += OP_loss
        # IOP_losses += IOP_loss
    
    # 计算 7层 平均损失
    total_loss =  total_losses / len(loss_dict_list)
    # OM_loss = OM_losses / len(loss_dict_list)
    # OP_loss = OP_losses / len(loss_dict_list) 
    # IOP_loss = IOP_losses / len(loss_dict_list) 

    return total_loss
  

def train_one_epoch(model, optimizer, epoch, train_dataloader, device, loss_fn, scaler, Metric, elnloss, l1_lambda, l2_lambda):
    """"
    model:             模型
    optimizer:         优化器
    epoch:             当前epoch
    train_dataloader:  训练数据集
    device:            设备
    loss_fn:           损失函数
    scaler:            梯度缩放器
    elnloss:           是否使用Elastic Net正则化
    l1_lambda:         l1正则化系数
    l2_lambda:         l2正则化系数
    """
    
    model.train()
    
    epoch_train_loss = 0.0
    epoch_OM_loss = 0.0
    epoch_OP_loss = 0.0
    epoch_IOP_loss = 0.0
    Metric_list = np.zeros((6, 4))

    # 使用 tqdm 包装 train_dataloader
    train_dataloader = tqdm(train_dataloader, desc=f" Training on Epoch :{epoch + 1}😀", leave=False)
    
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
                train_mean_loss = total_loss(pred, masks, loss_fn)  #  训练输出 7 个预测结果，6 个解码器输出和 1 个总输出。
                # if elnloss:
                #     # 添加Elastic Net正则化
                #     elastic_net_loss = model.elastic_net(l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                #     train_mean_loss = train_mean_loss + elastic_net_loss
                metrics = Metric.update(pred, masks)
                Metric_list += metrics
            
            # 是否使用辅助分类器
            elif isinstance(pred, tuple):
                heatmap, aux = pred

                # 主分支loss
                main_loss_dict = loss_fn(heatmap, masks)
                m_mean_loss = main_loss_dict['total_loss']
                m_OM_loss, m_OP_loss, m_IOP_loss = main_loss_dict['Organic matter'], main_loss_dict['Organic pores'], main_loss_dict['Inorganic pores']

                # 辅助分支loss
                aux_loss_dict = loss_fn(aux, masks)
                a_mean_loss = aux_loss_dict['total_loss']
                a_OM_loss, a_OP_loss, a_IOP_loss = aux_loss_dict['Organic matter'], aux_loss_dict['Organic pores'], aux_loss_dict['Inorganic pores']
                
                # 计算总损失：主分支损失*0.6 + 辅助分支损失*0.4
                train_mean_loss = m_mean_loss*0.6 + a_mean_loss*0.4
                OM_loss, OP_loss, IOP_loss = m_OM_loss*0.6 + a_OM_loss*0.4, m_OP_loss*0.6 + a_OP_loss*0.4, m_IOP_loss*0.6 + a_IOP_loss*0.4

                metrics = Metric.update(heatmap, masks)
                Metric_list += metrics

            else:
                loss_dict = loss_fn(pred, masks)
                train_mean_loss = loss_dict['total_loss']
                OM_loss, OP_loss, IOP_loss = loss_dict['Organic matter'], loss_dict['Organic pores'], loss_dict['Inorganic pores']
                # if elnloss:
                #     # 添加Elastic Net正则化
                #     elastic_net_loss = model.elastic_net(l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                #     train_mean_loss = train_mean_loss + elastic_net_loss

                metrics = Metric.update(pred, masks)
                Metric_list += metrics

        # 反向传播
        scaler.scale(train_mean_loss).backward()
      
        # 检查梯度是否包含inf或nan
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        epoch_train_loss += train_mean_loss.item()
        epoch_OM_loss += OM_loss.item()
        epoch_OP_loss += OP_loss.item()
        epoch_IOP_loss += IOP_loss.item()
    Metric_list /= len(train_dataloader)
        
    return epoch_train_loss, epoch_OM_loss, epoch_OP_loss, epoch_IOP_loss, Metric_list

def evaluate(model, device, data_loader, loss_fn, Metric, test:bool=False):
    """
    model:       模型
    device:      设备
    data_loader: 数据集
    loss_fn:     损失函数
    Metric:      指标
    """
    model.eval()
    if test:
        Metric_list = np.zeros((6, 4))
    else:
        Metric_list = np.zeros((6, 4))
    val_mean_loss = 0.0
    val_OM_loss = 0.0
    val_OP_loss = 0.0
    val_IOP_loss = 0.0


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
                    mean_loss = total_loss(pred_mask, masks, loss_fn)  #  训练输出 7 个预测结果，6 个解码器输出和 1 个总输出。
                    metrics = Metric.update(pred_mask, masks)
                    Metric_list += metrics

                # 是否使用辅助分类器
                elif isinstance(pred_mask, tuple):
                    heatmap, aux = pred_mask

                    # 主分支loss
                    main_loss_dict = loss_fn(heatmap, masks)
                    m_mean_loss = main_loss_dict['total_loss']
                    m_OM_loss, m_OP_loss, m_IOP_loss = main_loss_dict['Organic matter'], main_loss_dict['Organic pores'], main_loss_dict['Inorganic pores']

                    # 辅助分支loss
                    aux_loss_dict = loss_fn(aux, masks)
                    a_mean_loss = aux_loss_dict['total_loss']
                    a_OM_loss, a_OP_loss, a_IOP_loss = aux_loss_dict['Organic matter'], aux_loss_dict['Organic pores'], aux_loss_dict['Inorganic pores']
                    
                    # 计算总损失：主分支损失*0.6 + 辅助分支损失*0.4
                    mean_loss = m_mean_loss*0.6 + a_mean_loss*0.4
                    OM_loss, OP_loss, IOP_loss = m_OM_loss*0.6 + a_OM_loss*0.4, m_OP_loss*0.6 + a_OP_loss*0.4, m_IOP_loss*0.6 + a_IOP_loss*0.4

                    metrics = Metric.update(heatmap, masks)
                    Metric_list += metrics    

                else:
                    loss_dict = loss_fn(pred_mask, masks)
                    mean_loss = loss_dict['total_loss']
                    OM_loss, OP_loss, IOP_loss = loss_dict['Organic matter'], loss_dict['Organic pores'], loss_dict['Inorganic pores']

                    metrics = Metric.update(pred_mask, masks)
                    Metric_list += metrics    

            # 累加损失   # TODO : 2
            val_mean_loss += mean_loss.item()
            val_OM_loss += OM_loss.item()
            val_OP_loss += OP_loss.item()
            val_IOP_loss += IOP_loss.item()
    
    Metric_list /= len(val_dataloader)

    # TODO : 3
    return val_mean_loss,val_OM_loss,val_OP_loss,val_IOP_loss, Metric_list
