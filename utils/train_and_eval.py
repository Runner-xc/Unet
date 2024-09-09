from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch
import numpy as np
from metrics import *
import torch.nn.functional as F
"""
训练和验证
"""
def total_loss(inputs, target, loss_fn):
    """
    inputs: 预测值
    target: 真实值
    loss_fn: 损失函数
    """
    # 获取总的损失
    loss_list = [loss_fn(inputs[i], target) for i in range(len(inputs))]
    mean_loss_list = []
    OM_loss_list = []
    OP_loss_list = []
    IOP_loss_list = []

    # 遍历每一层损失
    for loss in loss_list: 
        OM_loss, OP_loss, IOP_loss, mean_loss = loss   # 使用 *rest 来捕获额外的损失
        mean_loss_list.append(mean_loss)
        OM_loss_list.append(OM_loss)
        OP_loss_list.append(OP_loss)
        IOP_loss_list.append(IOP_loss)
    
    # 计算平均损失
    train_loss = sum(mean_loss_list) / len(mean_loss_list)
    OM_loss = sum(OM_loss_list) / len(OM_loss_list)
    OP_loss = sum(OP_loss_list) / len(OM_loss_list)
    IOP_loss = sum(IOP_loss_list) / len(OM_loss_list)

    return OM_loss, OP_loss, IOP_loss, train_loss

def train_one_epoch(model, optimizer, epoch, train_dataloader, device, loss_fn, scaler):
    """"
    
    
    """
    
    model.train()
    
    train_loss = 0.0
    OM_loss = 0.0
    OP_loss = 0.0
    IOP_loss = 0.0

    # 使用 tqdm 包装 train_dataloader
    train_dataloader = tqdm(train_dataloader, desc=f" Training on Epoch :{epoch}😀", leave=False)
    
    for data in train_dataloader: 
        # 获取训练数据集的一个batch
        images, masks = data[0].to(device), data[1].to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 使用混合精度训练
        with autocast(device_type="cuda"):
            # 训练 + 计算loss
            pred_masks = model(images)
            train_OM_loss, train_OP_loss, train_IOP_loss, train_mean_loss = total_loss(pred_masks, masks, loss_fn)

        # 反向传播
        scaler.scale(train_mean_loss).backward()
        
        # 检查梯度是否包含inf或nan
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        scaler.step(optimizer)
        
        # 更新梯度缩放器
        scaler.update()
        
        # 更新进度条显示
        train_dataloader.update()

        # 累加损失
        train_loss += train_mean_loss.item()
        OM_loss += train_OM_loss.item()
        OP_loss += train_OP_loss.item()
        IOP_loss += train_IOP_loss.item()
    
    return train_loss, OM_loss, OP_loss, IOP_loss

def evaluate(model, device, data_loader, loss_fn, Metric):
    """
    
    """
    model.eval()
    Metric_list = np.zeros((4, 4))
    val_mean_loss = 0.0
    val_OM_loss = 0.0
    val_OP_loss = 0.0
    val_IOP_loss = 0.0


    with torch.no_grad():
        val_dataloader = tqdm(data_loader, desc=f"  Validating  😀", leave=False)
        for data in val_dataloader:
            images, masks =data[0].to(device), data[1].to(device)
            with autocast(device_type="cuda"):
                pred_masks = model(images)
                OM_loss, OP_loss, IOP_loss, mean_loss = loss_fn(pred_masks, masks)
                metrics = Metric.update(pred_masks, masks)
                Metric_list += metrics

            # 累加损失
            val_mean_loss += mean_loss.item()
            val_OM_loss += OM_loss.item()
            val_OP_loss += OP_loss.item()
            val_IOP_loss += IOP_loss.item()
    
    Metric_list /= len(val_dataloader)

    return val_OM_loss, val_OP_loss, val_IOP_loss, val_mean_loss, Metric_list