import sys
sys.path.append("..")  # 添加上级目录到路径中
from models import *
from tools import *
from monai.networks.nets import SegResNet, SwinUNETR, UNet
from config import get_config
import argparse
parser = argparse.ArgumentParser(description="Train a segmentation model with a flexible config system.")
parser.add_argument('--cfg', type=str, default="/mnt/e/VScode/WS-Hub/WS-UNet/UNet/cfg/swin_unet.yaml",help='Path to the .yaml config file.')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs, e.g. --opts TRAIN.EPOCHS 300 DATA.BATCH_SIZE 32",
    default=None,
    nargs='+',
)
args = parser.parse_args()
cfg = get_config(args)

def get_model_config(args):
    """
    根据模型名称获取对应的模型配置。
    
    :param model_name: 模型名称字符串
    :return: 模型配置字典
    """
    model_map = {

            # UNet 系列
            "u2net_full"                    : u2net_full_config(),
            "u2net_lite"                    : u2net_full_config(),
            # "unet"                          : UNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "unet"                          : UNet(spatial_dims=2,in_channels=3, out_channels=4, channels=(16*2, 32*2, 64*2, 128*2, 256*2), strides=(2, 2, 2, 2)),
            "att_unet"                      : Attention_UNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "res50_unet"                    : RES50_UNet(in_channels=3, num_classes=4),
            "swin_unet"                     : SwinUnet(cfg, num_classes=4),
            "swin_unetr"                    : SwinUNETR(spatial_dims=2, in_channels=3, out_channels=4),
            "seg_resnet"                    : SegResNet(spatial_dims=2, in_channels=3, out_channels=4),

            "ResD_unet"                     : ResD_UNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "aw_unet"                       : AWUNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "unetplusplus"                  : UnetPlusPlus(in_channels=3, num_classes=4, base_channel=32, deep_supervision=False),   
            "unet_3plus"                    : UNet_3Plus(in_channels=3, num_classes=4),
            # a_unet
            "a_unet"                        : A_UNet(in_channels=3, num_classes=4, base_channels=32,    p=args.dropout_p),
            "a_unetv2"                      : A_UNetV2(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "a_unetv3"                      : A_UNetV3(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "a_unetv4"                      : A_UNetV4(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "a_unetv5"                      : A_UNetV5(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "a_unetv6"                      : A_UNetV6(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),

            # m_unet
            "m_unet"                        : M_UNet(in_channels=3, num_classes=4, base_channels=32,    p=args.dropout_p),
            # "m_unetv2"                      : M_UNetV2(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p), 
            # "m_unetv3"                      : M_UNetV3(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),   

            # "ma_unet"                       : MAUNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "ds_dw_unet"                    : DeepSV_DW_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p),
            "ds_dw_unetv2"                  : DeepSV_DW_UNetV2(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p),

            # mamba
            "mamba_aunet"                   : Mamba_AUNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "mamba_aunetv2"                 : Mamba_AUNetV2(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "mamba_aunetv3"                 : Mamba_AUNetV3(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "mamba_aunetv4"                 : Mamba_AUNetV4(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            "mamba_aunetv5"                 : Mamba_AUNetV5(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p),
            
            # rdam_unet
            "rdam_unet"                     : RDAM_UNet(in_channels=3, num_classes=4, base_channels=32,  p=args.dropout_p, w=args.w),
            # "dwrdam_unet"                   : DWRDAM_UNet(in_channels=3, num_classes=4, base_channels=32,  p=0),
            # "dwrdam_unetv2"                 : DWRDAM_UNetV2(in_channels=3, num_classes=4, base_channels=32,  p=0),
            # 'dwrdam_unetv3'                 : DWRDAM_UNetV3(in_channels=3, num_classes=4, base_channels=32,  p=0),

            # 变体
            "aic_unet"                      : AICUNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p),
            "vm_unet"                       : VMUNet(input_channels=3, num_classes=4),
            "dc_unet"                       : DC_UNet(in_channels=3, num_classes=4, p=args.dropout_p),
            "att_unet_mamba"                : EM_UNetv3(in_channels=3, num_classes=4, p=args.dropout_p),

            # 其他架构
            "Segnet"                        : SegNet(num_classes=4, dropout_p=args.dropout_p),
            "pspnet"                        : PSPNet(classes=4, dropout=args.dropout_p, pretrained=False),
            "deeplabv3_resnet50"            : deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=4),
            "deeplabv3_resnet101"           : deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=4),
            # "deeplabv3_mobilenetv3_large"   : deeplabv3_mobilenetv3_large(aux=False, pretrain_backbone=False, num_classes=4)
        }
    model = model_map.get(args.model)
    if model is None:
        raise ValueError(f"Model {args.model} not found in model map.")
    return model

def get_lossfn(args):
    """
    根据损失函数名称获取对应的损失函数配置。
    
    :param loss_name: 损失函数名称字符串
    :return: 损失函数配置字典
    """ 
    loss_map = {
            'CrossEntropyLoss'  : CrossEntropyLoss(args.class_names),
            'DiceLoss'          : Diceloss(args.class_names),
            'DS_Dice'           : DS_Diceloss(args.class_names),
            'FocalLoss'         : Focal_Loss(args.class_names),
            'WDiceLoss'         : WDiceLoss(args.class_names),
            'DWDLoss'           : DWDLoss(args.class_names),
            'IoULoss'           : IOULoss(args.class_names),
            'ce_dice'           : CEDiceLoss(args.class_names),
            'dice_hd'           : AdaptiveSegLoss(4)
        }
    loss_fn = loss_map.get(args.loss_fn)
    if loss_fn is None:
        raise ValueError(f"Loss function {args.loss_fn} not found in loss map.")
    return loss_fn