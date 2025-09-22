# UNet 图像分割框架

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🔥 **项目简介**

这是一个专门针对**岩石图像语义分割**的深度学习框架，集成了20+种先进的分割网络架构，特别适用于页岩显微镜图像中有机质(OM)、有机孔隙(OP)、无机孔隙(IOP)的精确分割。

## ✨ 核心特性

- 🏗️ **模块化架构**: 清晰的代码组织，易于扩展和维护
- 🎯 **20+分割模型**: UNet系列、Transformer、Mamba等先进架构
- ⚙️ **统一配置系统**: YAML配置文件，支持命令行覆盖
- 📊 **完善的实验管理**: TensorBoard + SwanLab双重日志系统
- 🚀 **Web可视化界面**: Gradio驱动的在线推理平台
- 📈 **丰富的评估指标**: Dice、IoU、F1-Score等多维度评估
- 🔧 **便捷的开发工具**: 数据处理、配置管理、批量训练脚本
- 💾 **智能检查点**: 自动保存最佳模型，支持断点续训

## 📁 项目结构

```
UNet/
├── 📁 src/                     # 核心源代码
│   ├── models/                 # 分割模型实现
│   ├── utils/                  # 工具函数
│   ├── datasets/               # 数据处理
│   └── trainers/               # 训练器（预留）
├── 📁 configs/                 # 配置文件
│   ├── unet_default.yaml      # UNet基础配置
│   ├── swin_unet.yaml         # Swin-UNet配置
│   └── ...                    # 其他模型配置
├── 📁 examples/                # 示例和应用
│   ├── app.py                 # Gradio Web应用
│   └── predict.py             # 推理脚本
├── 📁 scripts/                 # 辅助脚本
│   ├── train_all.sh           # 批量训练
│   ├── prepare_data.py        # 数据准备
│   └── config_manager.py      # 配置管理
├── 📁 third_party/            # 第三方库
│   └── mamba/                 # Mamba实现
├── 📁 docs/                   # 文档
├── 📁 tests/                  # 测试用例
├── train.py                   # 主训练脚本
├── config.py                  # 配置管理
└── requirements.txt           # 依赖列表
```

## 🎯 训练模型

### 方式一：使用预定义配置文件

```bash
# 使用UNet + Dice Loss训练
python train_cfg.py --cfg cfg/unet_default.yaml

# 使用Swin-UNet训练
python train_cfg.py --cfg cfg/swin_unet.yaml

# 使用命令行覆盖配置
python train_cfg.py --cfg cfg/unet_default.yaml --opts TRAIN.EPOCHS 300 DATA.BATCH_SIZE 16
```

### 方式二：使用命令行参数

```bash
python train.py \
    --model unet \
    --loss_fn DiceLoss \
    --optimizer AdamW \
    --lr 5e-4 \
    --batch_size 8 \
    --end_epoch 200
```

### 主要训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型架构 | `swin_unet` |
| `--loss_fn` | 损失函数 | `DiceLoss` |
| `--optimizer` | 优化器 | `AdamW` |
| `--scheduler` | 学习率调度器 | `CosineAnnealingLR` |
| `--lr` | 学习率 | `5e-4` |
| `--batch_size` | 批次大小 | `8` |
| `--end_epoch` | 训练轮数 | `200` |

## 🔮 模型推理

### Web界面推理（推荐）

```bash
# 启动Gradio Web应用
python app.py
```

访问 `http://localhost:7860` 即可使用可视化界面进行推理。

### 命令行推理

```bash
# 单张图像推理
python predict.py \
    --model unet \
    --weights_path ./weights/unet/best_model.pth \
    --single True

# 批量推理（测试集）
python predict.py \
    --model unet \
    --weights_path ./weights/unet/best_model.pth \
    --data_path ./datasets/CSV/test_shale_256.csv
```

## 📊 支持的配置

### 损失函数
- `CrossEntropyLoss`: 交叉熵损失
- `DiceLoss`: Dice损失（推荐）
- `FocalLoss`: Focal损失
- `IoULoss`: IoU损失
- `ce_dice`: 交叉熵+Dice组合损失

### 优化器
- `AdamW`: 权重衰减Adam（推荐）
- `SGD`: 随机梯度下降
- `RMSprop`: RMSprop优化器

### 学习率调度器
- `CosineAnnealingLR`: 余弦退火调度器（推荐）
- `ReduceLROnPlateau`: 基于验证指标的调度器

## 📈 实验管理

### 日志可视化

```bash
# TensorBoard可视化
tensorboard --logdir ./training_results/tb_logs

# SwanLab在线可视化（自动启动）
# 训练时会自动记录到SwanLab平台
```

### 输出目录结构

```
📁 training_results/
├── 📁 weights/              # 模型权重
│   └── 📁 unet/
│       └── 📄 model_best_ep_50.pth
├── 📁 tb_logs/              # TensorBoard日志
├── 📁 runing_logs/          # 运行日志
└── 📁 mdf_args_log/         # 参数修改日志
```

## 🛠️ 高级功能

### 数据增强

项目支持智能数据增强，会根据数据集特点自动选择合适的增强策略：

- 几何变换：旋转、翻转、缩放
- 颜色变换：亮度、对比度、饱和度调整
- 噪声添加：高斯噪声、椒盐噪声
- 弹性变形：保持纹理特征的形变

### 混合精度训练

```bash
# 启用混合精度训练（默认启用）
python train.py --amp True
```

### 早停机制

```bash
# 设置早停patience（默认30个epoch）
python train.py --end_epoch 200  # 最大200轮，如果30轮无改善则早停
```

### 断点续训

```bash
# 从检查点恢复训练
python train.py --resume ./weights/unet/model_best_ep_50.pth
```

## 📋 实验结果示例

在页岩显微镜图像数据集上的分割结果：

| 模型 | mIoU | Dice | F1-Score | 备注 |
|------|------|------|----------|------|
| UNet | 0.756 | 0.823 | 0.818 | 基准模型 |
| Swin-UNet | 0.782 | 0.847 | 0.839 | Transformer架构 |
| UNet++ | 0.771 | 0.835 | 0.829 | 嵌套连接 |
| Attention UNet | 0.768 | 0.831 | 0.825 | 注意力机制 |

## 🔧 自定义开发

### 添加新模型

1. 在 `models/` 目录下创建新的模型文件
2. 在 `models/__init__.py` 中导入新模型
3. 在 `tools/cfg_loader.py` 中添加模型配置

### 添加新损失函数

1. 在 `tools/loss_fn.py` 中实现新的损失函数
2. 在 `get_lossfn()` 函数中添加对应映射

### 添加新数据集

1. 在 `tools/` 目录下创建对应的数据处理脚本
2. 参考 `SEM_DATA` 类的实现方式

## 📄 项目结构

```
UNet/
├── 📁 cfg/                   # 配置文件
├── 📁 models/                # 模型架构
├── 📁 tools/                 # 工具函数
├── 📁 mamba/                 # Mamba相关代码
├── 📁 datasets/              # 数据集
├── 📁 training_results/      # 训练结果
├── 📄 train.py              # 命令行训练脚本
├── 📄 train_cfg.py          # 配置文件训练脚本
├── 📄 predict.py            # 推理脚本
├── 📄 app.py                # Gradio Web应用
├── 📄 config.py             # 配置管理
└── 📄 requirements.txt      # 依赖列表
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请确保：

1. 代码符合项目风格
2. 添加适当的文档和注释
3. 提供测试用例
4. 更新相关文档


## 📜 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目的贡献：

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [Mamba](https://github.com/state-spaces/mamba)
- [Gradio](https://gradio.app/)
- [SwanLab](https://swanlab.cn/)

---
