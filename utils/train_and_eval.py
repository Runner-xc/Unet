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
    loss_list = [loss_fn(torch.softmax(model_output[i], dim=1), target) for i in range(len(model_output))]  
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
    model:             模型
    optimizer:         优化器
    epoch:             当前epoch
    train_dataloader:  训练数据集
    device:            设备
    loss_fn:           损失函数
    scaler:            梯度缩放器
    """
    
    model.train()
    
    epoch_train_loss = 0.0
    epoch_OM_loss = 0.0
    epoch_OP_loss = 0.0
    epoch_IOP_loss = 0.0

    # 使用 tqdm 包装 train_dataloader
    train_dataloader = tqdm(train_dataloader, desc=f" Training on Epoch :{epoch}😀", leave=False)
    
    for data in train_dataloader: 
        # 获取训练数据集的一个batch
        images, masks = data
        images, masks = images.to(device), masks.to(device)
        # 梯度清零
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(device_type="cuda"):
            # 训练 + 计算loss
            # pred_masks：list:(7, pred_mask)
            pred_masks = model(images)  #  训练输出 7 个预测结果，6 个解码器输出和 1 个总输出。
            OM_loss, OP_loss, IOP_loss, train_mean_loss = total_loss(pred_masks, masks, loss_fn)
           

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
        train_dataloader.set_postfix({"Loss": f"{train_mean_loss.item():.4f}"})
        train_dataloader.update()

        epoch_train_loss += train_mean_loss.item()
        epoch_OM_loss += OM_loss.item()
        epoch_OP_loss += OP_loss.item()
        epoch_IOP_loss += IOP_loss.item()
        
    return epoch_OM_loss, epoch_OP_loss, epoch_IOP_loss, epoch_train_loss / len(train_dataloader)

def evaluate(model, device, data_loader, loss_fn, Metric):
    """
    model:       模型
    device:      设备
    data_loader: 数据集
    loss_fn:     损失函数
    Metric:      指标
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
                pred_mask = model(images)         # 验证 模型输出 softmax 输出
                OM_loss, OP_loss, IOP_loss, mean_loss = loss_fn(pred_mask, masks)
           
                masks = masks.to(torch.int64)
                masks = masks.squeeze(1)
                metrics = Metric.update(pred_mask, masks)
                Metric_list += metrics    

            # 累加损失   # TODO : 2
            val_mean_loss += mean_loss.item()
            val_OM_loss += OM_loss.item()
            val_OP_loss += OP_loss.item()
            val_IOP_loss += IOP_loss.item()
    
    Metric_list /= len(val_dataloader)

    # TODO : 3
    return val_OM_loss, val_OP_loss, val_IOP_loss, val_mean_loss, Metric_list
