# Unet u2net
根据自身需求搭建Unet网络

## 项目文件结构
```
├── utils: 工具箱
├── SEM_Data.py:      自定义数据集读取相关代码
├── predict.py:       预测代码   (待实现)
├── train.py:         GPU 或 CPU 训练代码
├── transforms.py:    数据预处理相关代码  （待实现）
└── requirements.txt: 项目依赖
```
## 数据集结构
```
├── DATASETS
       ├── images: 该文件夹存放所有的图片
       ├── masks:  该文件夹存放所有的图片
       └── CSV:    该文件夹存放所有csv文件
            ├── DATASETS.csv: 总数据集的路径
            ├── TRAIN.csv:    训练数据集路径
            ├── VAL.csv:      验证数据集路径
            └── TEST.csv:     测试数据集路径
```
- 注意训练或者验证过程中，将`--data-path`指向`.csv`文件具体路径


## 代码已跑通，存在一些问题
- 1. 解决结果发散问题 （已解决：2024.09.22 15:10:12）
- 2. 增加调度器，动态调整学习率
- 3. 增加修改配置脚本
- 4. 增加 elastic net loss 正则化
- 5. 增加参数配置脚本

## 参数实验
| 模型 | 数据集大小 | epoch | lr | wd | l1_λ | l2_λ | optim | loss_fn | scheduler | train_loss | val_loss |
| :-----: | :---: | :---:| :---: | :---:  | :---: | :---: | :---: | :---: | :--- | :---: | :---: | 
| U-Net | 3000 | 150 | 0.0005 | 0 | 0.005 | 0 | AdamW | DiceLoss | <li>名称：CosineAnnealingLR<br><li>参数：t_max=20, eta_min=0.0001, last_epoch=5 | 收敛 | 很不稳定 |
| U-Net | 3000 | 150 | 0.0005 | 0 | 0.005 | 0 | AdamW | FocalLoss | <li>名称：CosineAnnealingLR<br><li>参数：t_max=20, eta_min=0.0001, last_epoch=5 | 不稳定 | 不稳定 |
| U-Net | 2000 | 150 | 0.0001 | 1e-5 | 0 | 0 | AdamW | FocalLoss | <li>名称：CosineAnnealingLR<br><li>参数：t_max=20, eta_min=0.0001, last_epoch=0 | 收敛 | 过拟合 |
| U-Net | 2000 | 150 | 0.0001 | 1e-4 | 0 | 0 | AdamW | FocalLoss | <li>名称：CosineAnnealingLR<br><li>参数：t_max=20, eta_min=0.0001, last_epoch=0 | 收敛 | 过拟合 |
| U-Net | 4000 | 150 | 0.001 | 1e-4 | 0 | 0 | AdamW | FocalLoss | <li>名称：CosineAnnealingLR<br><li>参数：t_max=20, eta_min=0.0001, last_epoch=0 | 下降，触发早停（15） | 持续下降 |
| U-Net | 12000 | 150 | 0.001 | 1e-4 | 0 | 0 | SGD<br>动量：0.9；wd：1e-4 | FocalLoss | <li>名称：ReduceLROnPlateau<br><li>参数：默认 | 收敛 | 收敛（135轮早停） |
| U-Net | 12000 | 150 | 0.001 | 1e-4 | 0 | 0 | SGD<br>动量：0.9；wd：1e-4 | FocalLoss | <li>名称：ReduceLROnPlateau<br><li>参数：默认 |  |  |

