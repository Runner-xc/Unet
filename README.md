# UNet 图像分割框架

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🔥 **专业级图像分割框架**

这是一个专门针对**岩石图像语义分割**的深度学习框架，集成了20+种先进的分割网络架构，特别适用于页岩显微镜图像中有机质(OM)、有机孔隙(OP)、无机孔隙(IOP)的精确分割。框架经过重构，具有清晰的代码结构、统一的配置管理和完善的工具链。

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

## 🎯 支持的模型架构

### 🏛️ 经典UNet系列
- **UNet**: 经典U型网络架构
- **UNet++**: 嵌套U型网络，密集跳跃连接
- **UNet-3Plus**: 全尺度跳跃连接
- **Attention UNet**: 注意力机制增强版
- **ResNet-UNet**: ResNet骨干网络版本

### 🌟 现代Transformer架构
- **Swin-UNet**: Swin Transformer + UNet
- **Vision Mamba UNet**: 基于Mamba的高效架构

### 🔬 专业分割网络
- **AIC-UNet**: 自适应通道注意力UNet
- **DC-UNet**: 深度可分离卷积UNet
- **EM-UNet**: 高效多尺度UNet
- **DeepLabV3**: 带ASPP的语义分割
- **PSPNet**: 金字塔场景解析网络
- **SegNet**: 编码器-解码器架构
- **U2-Net**: 显著性检测网络

## 🚀 快速开始

### 1️⃣ 环境配置

```bash
# 克隆项目
git clone https://github.com/your-username/UNet-Segmentation.git
cd UNet-Segmentation

# 创建虚拟环境
conda create -n unet python=3.10
conda activate unet

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 数据准备

项目期望的数据格式：

```
datasets/
├── CSV/
│   ├── shale_256.csv          # 数据索引文件
│   └── shale_512.csv
├── images/                    # 原始图像
│   ├── 256/
│   └── 512/
└── masks/                     # 分割标签
    ├── 256/
    └── 512/
```

**准备数据：**

```bash
# 自动数据划分
python scripts/prepare_data.py --data_path datasets/CSV/shale_256.csv

# 自定义划分比例
python scripts/prepare_data.py --train_ratio 0.7 --val_ratio 0.2 --force_split
```

### 3️⃣ 模型训练

**快速开始（推荐）：**

```bash
# 使用默认配置训练UNet
python train.py --cfg configs/unet_default.yaml

# 训练Swin-UNet
python train.py --cfg configs/swin_unet.yaml

# 使用命令行覆盖配置
python train.py --cfg configs/unet_default.yaml --opts TRAIN.EPOCHS 300 DATA.BATCH_SIZE 16
```

**批量训练：**

```bash
# 训练多个模型
bash scripts/train_all.sh
```

**自定义配置：**

```bash
# 创建新的配置文件
python scripts/config_manager.py --create my_model --output configs/my_model.yaml

# 查看所有可用配置
python scripts/config_manager.py --list
```

### 4️⃣ 模型推理

**Web界面（推荐）：**

```bash
# 启动Gradio应用
cd examples
python app.py

# 访问 http://localhost:7860
```

**命令行推理：**

```bash
# 单张图像推理
cd examples
python predict.py \
    --model unet \
    --weights_path ../training_results/weights/unet/model_best.pth \
    --single True

# 批量推理
python predict.py \
    --model unet \
    --weights_path ../training_results/weights/unet/model_best.pth \
    --data_path ../datasets/CSV/test_shale_256.csv
```

## ⚙️ 配置说明

### 核心配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL.NAME` | 模型架构名称 | `unet` |
| `TRAIN.OPTIMIZER.BASE_LR` | 学习率 | `5e-4` |
| `DATA.BATCH_SIZE` | 批次大小 | `8` |
| `TRAIN.EPOCHS` | 训练轮数 | `200` |
| `TRAIN.LOSS.NAME` | 损失函数 | `DiceLoss` |
| `TRAIN.SCHEDULER.NAME` | 学习率调度器 | `CosineAnnealingLR` |

### 损失函数选择

- `DiceLoss`: 适用于分割任务，处理类别不平衡
- `CrossEntropyLoss`: 经典分类损失
- `FocalLoss`: 处理困难样本
- `IoULoss`: 直接优化IoU指标
- `ce_dice`: 交叉熵+Dice组合损失

### 优化器配置

- `AdamW`: 权重衰减Adam（推荐）
- `SGD`: 随机梯度下降
- `RMSprop`: RMSprop优化器

## 📈 性能基准

在页岩显微镜图像数据集上的性能表现：

| 模型 | mIoU | Dice | F1-Score | 参数量 | 训练时间 |
|------|------|------|----------|--------|----------|
| UNet | 0.756 | 0.823 | 0.818 | 7.8M | 2.5h |
| UNet++ | 0.771 | 0.835 | 0.829 | 9.1M | 3.2h |
| Swin-UNet | 0.782 | 0.847 | 0.839 | 27.2M | 4.1h |
| Attention UNet | 0.768 | 0.831 | 0.825 | 8.7M | 2.8h |
| VM-UNet | 0.775 | 0.842 | 0.836 | 32.5M | 3.9h |

*测试环境: RTX 3090, batch_size=8, 200 epochs*

## 🔧 开发指南

### 添加新模型

1. 在 `src/models/` 创建模型文件
2. 在 `src/models/__init__.py` 中导入
3. 在 `src/utils/cfg_loader.py` 中注册
4. 创建对应的配置文件

### 添加新损失函数

1. 在 `src/utils/loss_fn.py` 实现
2. 在 `get_lossfn()` 函数中注册

### 自定义数据集

1. 参考 `src/utils/data_split.py` 中的 `SEM_DATA` 类
2. 实现自己的数据加载逻辑
3. 在配置文件中指定数据路径

## 📧 联系方式

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/UNet-Segmentation/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/UNet-Segmentation/discussions)

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目的贡献：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Transformers](https://huggingface.co/transformers/) - Transformer模型
- [Mamba](https://github.com/state-spaces/mamba) - 状态空间模型
- [Gradio](https://gradio.app/) - Web界面框架
- [SwanLab](https://swanlab.cn/) - 实验管理平台
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 可视化工具

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！