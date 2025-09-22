#!/usr/bin/env python3
"""
UNet分割网络训练脚本
支持YAML配置文件和命令行参数两种配置方式
"""

import torch
import datetime
import argparse
import os
import time
import math
from pathlib import Path
from typing import Union, List, Dict

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import swanlab
import yaml

from rich.console import Console
from tabulate import tabulate

# 导入自定义模块
from config import get_config
from src.utils.cfg_loader import get_model_config, get_lossfn
from src.utils.data_split_cfg import small_data_split_to_train_val_test_cfg, data_split_to_train_val_test_cfg
from src.utils.data_split import SEM_DATA
from src.utils.model_initial import init_weights_2d
from src.utils.metrics import Metrics
from src.utils import transforms

color = Console()

class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(base_size),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(base_size),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

# 表情符号配置
PARAM_ICONS = {
    'model': '📦', 'lr': '📚', 'wd': '⚖️', 'dropout': '☔',
    'l1_lambda': 'λ¹', 'l2_lambda': 'λ²', 'scheduler': '🔄',
    'loss_fn': '💥', 'best_epoch': '🏆', 'time': '🕒', 'cost': '⏳'
}

def format_epoch_header(epoch, end_epoch):
    return f"\n🚀✨========= Epoch {epoch+1}/{end_epoch} =========✨🚀\n"

def build_params_block(cfg):
    return (
        f"{PARAM_ICONS['model']} model : {cfg.MODEL.NAME}\n"
        f"{PARAM_ICONS['lr']} lr : {cfg.TRAIN.OPTIMIZER.BASE_LR}\n"
        f"{PARAM_ICONS['wd']} wd : {cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}\n"
        f"{PARAM_ICONS['dropout']} dropout : {cfg.MODEL.DROPOUT_P}\n"
        f"{PARAM_ICONS['l1_lambda']} l1_lambda : {cfg.TRAIN.LOSS.L1_LAMBDA}\n"
        f"{PARAM_ICONS['l2_lambda']} l2_lambda : {cfg.TRAIN.LOSS.L2_LAMBDA}\n"
        f"{PARAM_ICONS['loss_fn']} loss_fn : {cfg.TRAIN.LOSS.NAME}\n"
        f"{PARAM_ICONS['scheduler']} scheduler : {cfg.TRAIN.SCHEDULER.NAME}\n"
    )

def setup_model(cfg, device):
    """设置模型"""
    model = get_model_config(cfg).to(device)
    model.apply(init_weights_2d)
    return model

def setup_optimizer(cfg, model):
    """设置优化器"""
    if cfg.TRAIN.OPTIMIZER.NAME == 'AdamW':
        return AdamW(
            model.parameters(), 
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR, 
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY, 
            betas=cfg.TRAIN.OPTIMIZER.ADAMW_BETAS
        )
    elif cfg.TRAIN.OPTIMIZER.NAME == 'SGD':
        return SGD(
            model.parameters(),
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            momentum=cfg.TRAIN.OPTIMIZER.SGD_MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    elif cfg.TRAIN.OPTIMIZER.NAME == 'RMSprop':
        return RMSprop(
            model.parameters(),
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.TRAIN.OPTIMIZER.NAME} not implemented.")

def setup_scheduler(cfg, optimizer, train_loader):
    """设置学习率调度器"""
    if cfg.TRAIN.SCHEDULER.NAME == 'CosineAnnealingLR':
        num_batches_per_epoch = len(train_loader)
        warmup_steps = cfg.TRAIN.SCHEDULER.WARMUP_EPOCHS * num_batches_per_epoch
        Tmax_steps = cfg.TRAIN.SCHEDULER.COSINE_T_MAX * num_batches_per_epoch
        lr_initial = cfg.TRAIN.OPTIMIZER.BASE_LR
        eta_min = cfg.TRAIN.SCHEDULER.COSINE_ETA_MIN
        
        return LambdaLR(
            optimizer, 
            lr_lambda=lambda step: (
                (step / warmup_steps) if step < warmup_steps
                else (eta_min + (lr_initial - eta_min) * 
                     (1 + math.cos(math.pi * (step - warmup_steps) / Tmax_steps)) / 2) / lr_initial
            )
        )
    elif cfg.TRAIN.SCHEDULER.NAME == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=cfg.TRAIN.SCHEDULER.PLATEAU_FACTOR, 
            patience=cfg.TRAIN.SCHEDULER.PLATEAU_PATIENCE
        )
    else:
        raise NotImplementedError(f"Scheduler {cfg.TRAIN.SCHEDULER.NAME} not implemented.")

def setup_data_loaders(cfg):
    """设置数据加载器"""
    train_transforms = SODPresetTrain(
        cfg.DATA.IMG_SIZE, 
        cfg.DATA.IMG_SIZE, 
        mean=cfg.DATA.TRANSFORMS.MEAN, 
        std=cfg.DATA.TRANSFORMS.STD
    )
    eval_transforms = SODPresetEval(
        cfg.DATA.IMG_SIZE, 
        mean=cfg.DATA.TRANSFORMS.MEAN, 
        std=cfg.DATA.TRANSFORMS.STD
    )

    # 数据集划分
    if cfg.DATA.NUM_SMALL_DATA:
        train_csv, val_csv, _ = small_data_split_to_train_val_test_cfg(cfg)
    else:
        train_csv, val_csv, _ = data_split_to_train_val_test_cfg(cfg)
    
    train_dataset = SEM_DATA(train_csv, transforms=train_transforms)
    val_dataset = SEM_DATA(val_csv, transforms=eval_transforms)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.DATA.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.SYSTEM.NUM_WORKERS, 
        pin_memory=cfg.SYSTEM.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.DATA.VALID_BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.SYSTEM.NUM_WORKERS, 
        pin_memory=cfg.SYSTEM.PIN_MEMORY
    )

    return train_loader, val_loader

def setup_logging(cfg, detailed_time_str):
    """设置日志系统"""
    # 创建输出目录
    base_output_path = f"{cfg.OUTPUT}/{cfg.MODEL.NAME}/{cfg.TRAIN.LOSS.NAME}-{cfg.TRAIN.SCHEDULER.NAME}"
    if cfg.TRAIN.LOSS.ELN_LOSS:
        experiment_folder = f"{cfg.TRAIN.OPTIMIZER.NAME}-lr_{cfg.TRAIN.OPTIMIZER.BASE_LR}-l1_{cfg.TRAIN.LOSS.L1_LAMBDA}-l2_{cfg.TRAIN.LOSS.L2_LAMBDA}"
    else:
        experiment_folder = f"{cfg.TRAIN.OPTIMIZER.NAME}-lr_{cfg.TRAIN.OPTIMIZER.BASE_LR}-wd_{cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}"
    
    output_dir = os.path.join(base_output_path, experiment_folder, detailed_time_str)
    weights_path = os.path.join(output_dir, "weights")
    logs_path = os.path.join(output_dir, "logs")
    
    if cfg.LOG.SAVE_FLAG:
        os.makedirs(weights_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

    # TensorBoard
    writer = None
    if cfg.LOG.USE_TENSORBOARD and cfg.LOG.SAVE_FLAG:
        writer = SummaryWriter(log_dir=os.path.join(logs_path, "tb"))

    # SwanLab
    swanlab_active = False
    if cfg.LOG.USE_SWANLAB and cfg.LOG.SAVE_FLAG:
        swanlab.init(
            project="UNet-Segmentation",
            experiment_name=f"{cfg.MODEL.NAME}-{cfg.TAG}-{detailed_time_str}",
            config=yaml.safe_load(cfg.dump())
        )
        swanlab_active = True

    return output_dir, weights_path, logs_path, writer, swanlab_active

def train_one_epoch(model, optimizer, train_loader, device, loss_fn, scaler, metrics, scheduler, cfg):
    """训练一个epoch"""
    model.train()
    train_losses = [0.0] * (len(cfg.DATA.CLASS_NAMES) + 1)  # 每个类别 + 总损失
    metric_list = [0.0] * 6  # 6个指标
    
    for data in train_loader:
        images, masks = data[0][0].to(device), data[0][1].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                losses = loss_fn(outputs, masks.long().squeeze(1))
        else:
            outputs = model(images)
            losses = loss_fn(outputs, masks.long().squeeze(1))
        
        # 计算总损失
        if isinstance(losses, list):
            total_loss = sum(losses)
            for i, loss in enumerate(losses):
                train_losses[i] += loss.item()
        else:
            total_loss = losses
            train_losses[-1] += total_loss.item()
        
        # 添加正则化
        if cfg.TRAIN.LOSS.ELN_LOSS:
            l1_reg = sum(param.abs().sum() for param in model.parameters())
            l2_reg = sum((param**2).sum() for param in model.parameters())
            total_loss += cfg.TRAIN.LOSS.L1_LAMBDA * l1_reg + cfg.TRAIN.LOSS.L2_LAMBDA * l2_reg
        
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        # 更新调度器
        if hasattr(scheduler, 'step') and cfg.TRAIN.SCHEDULER.NAME == 'CosineAnnealingLR':
            scheduler.step()
        
        # 计算指标
        with torch.no_grad():
            pred_mask = torch.softmax(outputs, dim=1)
            batch_metrics = metrics.update(pred_mask, masks.long().squeeze(1))
            for i in range(6):
                metric_list[i] += batch_metrics[i]
    
    return train_losses, metric_list

def validate(model, val_loader, device, loss_fn, metrics, cfg):
    """验证"""
    model.eval()
    val_losses = [0.0] * (len(cfg.DATA.CLASS_NAMES) + 1)
    metric_list = [0.0] * 6
    
    with torch.no_grad():
        for data in val_loader:
            images, masks = data[0][0].to(device), data[0][1].to(device)
            
            outputs = model(images)
            losses = loss_fn(outputs, masks.long().squeeze(1))
            
            if isinstance(losses, list):
                for i, loss in enumerate(losses):
                    val_losses[i] += loss.item()
            else:
                val_losses[-1] += losses.item()
            
            pred_mask = torch.softmax(outputs, dim=1)
            batch_metrics = metrics.update(pred_mask, masks.long().squeeze(1))
            for i in range(6):
                metric_list[i] += batch_metrics[i]
    
    return val_losses, metric_list

def main(cfg):
    """主训练函数"""
    color.print("🚀======= 开始训练 =======🚀")
    color.print(cfg)
    
    # 设置设备和时间戳
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else "cpu")
    detailed_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 设置模型、优化器等组件
    model = setup_model(cfg, device)
    model_info = str(summary(model, (1, 3, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)))
    
    train_loader, val_loader = setup_data_loaders(cfg)
    optimizer = setup_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optimizer, train_loader)
    loss_fn = get_lossfn(cfg)
    scaler = torch.amp.GradScaler(enabled=cfg.TRAIN.AMP)
    metrics = Metrics(cfg.DATA.CLASS_NAMES)
    
    # 设置日志
    output_dir, weights_path, logs_path, writer, swanlab_active = setup_logging(cfg, detailed_time_str)
    
    # 训练循环
    start_epoch = cfg.TRAIN.START_EPOCH
    best_loss = float('inf')
    best_epoch = -1
    patience = 0
    
    # 断点续传
    if cfg.MODEL.RESUME:
        checkpoint = torch.load(cfg.MODEL.RESUME, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['epoch']
        color.print(f"🌐Resume from epoch: [blue]{start_epoch}[/blue]")
    
    initial_time = time.time()
    
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        epoch_start_time = time.time()
        
        # 训练
        color.print(format_epoch_header(epoch, cfg.TRAIN.EPOCHS))
        color.print(f"🌈 ---- Training ---- 🌈")
        
        train_losses, train_metrics_list = train_one_epoch(
            model, optimizer, train_loader, device, loss_fn, scaler, metrics, scheduler, cfg
        )
        
        avg_train_losses = [l / len(train_loader) for l in train_losses]
        avg_train_total_loss = avg_train_losses[-1]
        
        # 验证
        if epoch % cfg.TRAIN.EVAL_INTERVAL == 0 or epoch == cfg.TRAIN.EPOCHS - 1:
            color.print(f"🌈 ---- Validation ---- 🌈")
            
            val_losses, val_metrics_list = validate(model, val_loader, device, loss_fn, metrics, cfg)
            avg_val_losses = [l / len(val_loader) for l in val_losses]
            avg_val_total_loss = avg_val_losses[-1]
            
            current_lr = optimizer.param_groups[0]['lr']
            epoch_end_time = time.time()
            
            # 日志记录
            if writer:
                loss_names = cfg.DATA.CLASS_NAMES + ['total_loss']
                for i, name in enumerate(loss_names):
                    writer.add_scalar(f'Loss_train/{name}', avg_train_losses[i], epoch)
                    writer.add_scalar(f'Loss_val/{name}', avg_val_losses[i], epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            if swanlab_active:
                log_data = {
                    **{f"train/{name}_loss": loss for name, loss in zip(cfg.DATA.CLASS_NAMES + ['total'], avg_train_losses)},
                    **{f"val/{name}_loss": loss for name, loss in zip(cfg.DATA.CLASS_NAMES + ['total'], avg_val_losses)},
                    "learning_rate": current_lr,
                    "epoch": epoch + 1
                }
                swanlab.log(log_data)
            
            # 控制台输出
            metrics_table_header = ['Metrics'] + cfg.DATA.CLASS_NAMES + ['Total']
            metrics_table_left = ['mIoU', 'Dice', 'F1_scores', 'Recall', 'Precision', 'Accuracy']
            metrics_table = [
                [name] + [f"{val:.4f}" for val in val_metrics_list[i]]
                for i, name in enumerate(metrics_table_left)
            ]
            
            if avg_val_total_loss < best_loss:
                best_loss = avg_val_total_loss
                best_epoch = epoch + 1
            
            training_info_block = (
                f"{PARAM_ICONS['time']} time: {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}\n"
                f"{PARAM_ICONS['cost']} epoch_cost: {(epoch_end_time - epoch_start_time):.2f}s\n"
                f"{PARAM_ICONS['best_epoch']} Best Epoch: {best_epoch} (Loss: {best_loss:.4f})\n"
                f"🚀 Current LR: [green]{current_lr:.7f}[/green]"
            )
            
            console_output = (
                "\n========= 训练配置 =========\n" +
                build_params_block(cfg) +
                "\n========= 性能指标 (Validation) =========\n" +
                tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid') + "\n" +
                "\n========= 训练状态 =========\n" +
                training_info_block
            )
            
            color.print(console_output)
            
            # 保存运行日志
            if cfg.LOG.SAVE_FLAG:
                log_file_path = os.path.join(logs_path, "run_log.txt")
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(console_output + "\n\n")
            
            # 保存最佳模型
            if avg_val_total_loss < best_loss:
                best_loss = avg_val_total_loss
                best_epoch = epoch + 1
                patience = 0
                
                if cfg.LOG.SAVE_FLAG:
                    save_file = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': best_epoch,
                        'best_loss': best_loss,
                        'cfg': cfg.dump(),
                        'model_info': model_info
                    }
                    
                    # 删除旧的最佳模型
                    for f in Path(weights_path).glob("model_best_*.pth"):
                        f.unlink()
                    
                    # 保存新的最佳模型
                    best_model_path = f"{weights_path}/model_best_ep_{best_epoch}.pth"
                    torch.save(save_file, best_model_path)
                    color.print(f"✨ Best model saved at epoch: {best_epoch} ✨ with loss: {best_loss:.4f}")
            else:
                patience += 1
        
        # 早停检查
        if patience >= cfg.TRAIN.EARLY_STOP_PATIENCE:
            color.print(f'✅ Early stopping triggered after {patience} epochs.')
            break
    
    # 训练结束
    if writer:
        writer.close()
    if swanlab_active:
        swanlab.finish()
    
    total_time = time.time() - initial_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    color.print(f"✅ Training finished. ⏳ Total time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a segmentation model with YAML config.")
    parser.add_argument('--cfg', type=str, default="configs/unet_default.yaml", 
                       help='Path to the .yaml config file.')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    
    args = parser.parse_args()
    cfg = get_config(args)
    main(cfg)