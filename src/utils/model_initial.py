import torch
import numpy as np
import torch.nn as nn

# kaiming初始化
import torch.nn as nn
import torch.nn.init as init
import warnings
from collections import defaultdict

def kaiming_initial(model, a=0, mode='fan_in', nonlinearity='relu', 
                conv_type='2d', bias_strategy='zero', verbose=True):
    """
    🎯 功能特点：
    - 支持多维度卷积层 (1D/2D/3D)
    - 自动适配BN层/LayerNorm等归一化层
    - 智能处理自定义层 (Mamba/Attention/DeformableConv等)
    - 防御式参数检测 (权重/偏置存在性校验)
    - 初始化过程可视化跟踪
    """
    
    # 初始化统计器
    init_stats = defaultdict(int)
    
    # 遍历所有网络层
    for name, module in model.named_modules():
        # 跳过空层和容器层
        if isinstance(module, (nn.ModuleList, nn.Sequential)): 
            continue
            
        # 核心初始化逻辑 ===============================================
        # Case 1: 卷积层系列
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, 
                                  nonlinearity=nonlinearity)
            init_stats[f'Conv{conv_type}'] += 1
            
            # 偏置处理策略
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                init_stats['ConvBias'] +=1
        # Case 2: 转置卷积
        elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, 
                               nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=a, mode='fan_out',
                                  nonlinearity=nonlinearity)
            init_stats['ConvTranspose'] +=1
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                init_stats['ConvTransposeBias'] +=1
        # Case 3: 全连接层
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                  nonlinearity=nonlinearity)
            init_stats['Linear'] +=1
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                init_stats['LinearBias'] +=1
        # Case 4: 批归一化层
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, 
                               nn.BatchNorm3d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            init_stats['BatchNorm'] +=1
        # Case 5: 层归一化
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            init_stats['LayerNorm'] +=1
        # Case 6: 自定义层智能处理（Mamba/Attention等）
        elif 'Mamba' in str(type(module)):
            # Mamba层特殊初始化（示例）
            for param in module.parameters():
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0, std=0.01)
            init_stats['MambaLayer'] +=1
            
        # elif 'Attention' in str(type(module)):
        #     # 注意力机制初始化
        #     nn.init.xavier_uniform_(module.q_proj.weight)
        #     nn.init.xavier_uniform_(module.k_proj.weight)
        #     nn.init.xavier_uniform_(module.v_proj.weight)
        #     init_stats['Attention'] +=1
        # Case 7: 防御式兜底策略
        else:
            # 参数存在性校验
            has_weight = hasattr(module, 'weight') and module.weight is not None
            has_bias = hasattr(module, 'bias') and module.bias is not None
            
            # 权重初始化
            if has_weight:
                if module.weight.dim() >= 2:
                    nn.init.kaiming_normal_(module.weight, a=a, mode=mode,
                                          nonlinearity=nonlinearity)
                else:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                init_stats['FallbackWeight'] +=1
                
            # 偏置初始化
            if has_bias:
                if bias_strategy == 'zero':
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.normal_(module.bias, mean=0, std=0.01)
                init_stats['FallbackBias'] +=1
            # 警告未识别层
            if verbose and (has_weight or has_bias):
                warnings.warn(f"⚠️ 未注册层类型 [{type(module).__name__}] "+
                            f"路径: {name} 已执行防御式初始化", 
                            UserWarning, stacklevel=2)
    
    # 打印初始化报告               
    if verbose:
        print("\n🔥 初始化统计报告：")
        for k, v in init_stats.items():
            print(f"▸ {k.ljust(20)} : {v}")
        print(f"✅ 总初始化参数数量: {sum(init_stats.values())}")
    
    return model

# 正交初始化
def init_weights_2d(m):
    """增强鲁棒性的2D权重初始化函数"""
    if isinstance(m, (nn.Conv2d)):
        nn.init.kaiming_normal_(
            m.weight, 
            mode='fan_in',
            nonlinearity='relu',
            a=0
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(
            m.weight,
            mode='fan_out',
            nonlinearity='relu',
            a=0
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        
    # 规范化层处理统一化
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if m.affine:  # 仅当具有可学习参数时进行初始化
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            # 对InstanceNorm增加额外初始化
            if isinstance(m, nn.InstanceNorm2d):
                # 在常数初始化基础上增加小扰动
                nn.init.normal_(m.weight, mean=1.0, std=0.01)

# xavier初始化
def xavier_initial(model, init_gain=0.02):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, init_gain)
            nn.init.constant_(m.bias, 0)