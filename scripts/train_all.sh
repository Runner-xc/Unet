#!/bin/bash
"""
快速训练脚本
提供常用的训练配置模板
"""

echo "🚀 UNet分割网络 - 快速训练脚本"
echo "================================="

# 基础UNet训练
echo "📦 训练基础UNet模型..."
python train.py --cfg configs/unet_default.yaml

# Swin-UNet训练  
echo "🌟 训练Swin-UNet模型..."
python train.py --cfg configs/swin_unet.yaml

# 高级模型训练
echo "🔥 训练高级模型..."
for model in "att_unet_default" "unetplusplus_default" "vm_unet_default"; do
    echo "   正在训练: $model"
    python train.py --cfg configs/${model}.yaml
done

echo "✅ 所有模型训练完成！"