import sys
sys.path.append("..")  # 添加上级目录到路径中
from src.models.utils import *

def get_model_cfg(cfg):
    """
    根据配置对象(cfg)获取对应的模型实例。
    
    :param cfg: yacs配置节点对象
    :return: 初始化后的PyTorch模型
    """
    model_name = cfg.MODEL.NAME
    dropout_p = cfg.MODEL.DROPOUT_P
    # 类别数从config中获取，通常是类别列表长度+1（背景类）
    num_classes = len(cfg.DATA.CLASS_NAMES) + 1 
    
    # 假设模型的输入通道数固定为3
    in_channels = 3 
    # 假设U-Net系列的基础通道数可以配置，如果需要的话
    base_channels = cfg.MODEL.get('BASE_CHANNELS', 32) # 使用.get()方法，如果没定义就用默认值32

    model_map = {
            # UNet 系列
            "u2net_full"                    : u2net_full_config(out_ch=num_classes),
            "u2net_lite"                    : u2net_lite_config(out_ch=num_classes),
            "unet"                          : UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "att_unet"                      : Attention_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "res50_unet"                    : RES50_UNet(in_channels=in_channels, num_classes=num_classes),
            "res18_unet"                    : RES18_UNet(in_channels=in_channels, num_classes=num_classes),
            "swin_unet"                     : SwinUnet(config=cfg, num_classes=num_classes),

            "ResD_unet"                     : ResD_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "aw_unet"                       : AWUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "unetplusplus"                  : UnetPlusPlus(in_channels=in_channels, num_classes=num_classes, base_channel=base_channels, deep_supervision=False),   
            "unet_3plus"                    : UNet_3Plus(in_channels=in_channels, num_classes=num_classes),
            # a_unet
            "a_unet"                        : A_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "a_unetv2"                      : A_UNetV2(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "a_unetv3"                      : A_UNetV3(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "a_unetv4"                      : A_UNetV4(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "a_unetv5"                      : A_UNetV5(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "a_unetv6"                      : A_UNetV6(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),

            # m_unet
            "m_unet"                        : M_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "m_unetv2"                      : M_UNetV2(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p), 
            "m_unetv3"                      : M_UNetV3(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),   

            "ma_unet"                       : MAUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "ds_dw_unet"                    : DeepSV_DW_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "ds_dw_unetv2"                  : DeepSV_DW_UNetV2(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),

            # mamba
            "mamba_aunet"                   : Mamba_AUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "mamba_aunetv2"                 : Mamba_AUNetV2(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "mamba_aunetv3"                 : Mamba_AUNetV3(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "mamba_aunetv4"                 : Mamba_AUNetV4(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "mamba_aunetv5"                 : Mamba_AUNetV5(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            'rm_unet'                       : RMUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            # rdam_unet
            "rdam_unet"                     : RDAM_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "dwrdam_unet"                   : DWRDAM_UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=0),
            "dwrdam_unetv2"                 : DWRDAM_UNetV2(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=0),
            'dwrdam_unetv3'                 : DWRDAM_UNetV3(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=0),

            # 变体
            "aic_unet"                      : AICUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels, p=dropout_p),
            "vm_unet"                       : VMUNet(input_channels=in_channels, num_classes=num_classes, dropout_r=dropout_p),
            "dc_unet"                       : DC_UNet(in_channels=in_channels, num_classes=num_classes, p=dropout_p),
            "em_unet"                       : EM_UNet(in_channels=in_channels, num_classes=num_classes, p=dropout_p),
            "em_unetv2"                     : EM_UNetv2(in_channels=in_channels, num_classes=num_classes, p=dropout_p),
            "att_unet_mamba"                : EM_UNetv3(in_channels=in_channels, num_classes=num_classes, p=dropout_p),

            # 其他架构
            "Segnet"                        : SegNet(num_classes=num_classes, dropout_p=dropout_p),
            "pspnet"                        : PSPNet(classes=num_classes, dropout=dropout_p, pretrained=False),
            "deeplabv3_resnet50"            : deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=num_classes),
            "deeplabv3_resnet101"           : deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=num_classes),
        }
        
    model = model_map.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found in model map.")
    return model

def get_lossfn_cfg(cfg):
    """
    根据配置对象(cfg)获取对应的损失函数实例。
    
    :param cfg: yacs配置节点对象
    :return: 损失函数实例
    """ 
    loss_name = cfg.TRAIN.LOSS.NAME
    class_names = cfg.DATA.CLASS_NAMES # 从config获取类别名
    num_classes = len(class_names) + 1 # 加上背景类

    loss_map = {
            'CrossEntropyLoss'  : CrossEntropyLoss(class_names),
            'DiceLoss'          : Diceloss(class_names),
            'DS_Dice'           : DS_Diceloss(class_names),
            'FocalLoss'         : Focal_Loss(class_names),
            'WDiceLoss'         : WDiceLoss(class_names),
            'DWDLoss'           : DWDLoss(class_names),
            'IoULoss'           : IOULoss(class_names),
            'ce_dice'           : CEDiceLoss(class_names),
            'dice_hd'           : AdaptiveSegLoss(num_classes)
        }
        
    loss_fn = loss_map.get(loss_name)
    if loss_fn is None:
        raise ValueError(f"Loss function '{loss_name}' not found in loss map.")
    return loss_fn