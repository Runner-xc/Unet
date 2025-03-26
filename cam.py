"""
unet
"""
import os
import torch
from model.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large
from model.pspnet import PSPNet
from model.Segnet import SegNet
from model.u2net import u2net_full_config, u2net_lite_config
from model.unet import UNet, ResD_UNet
from model.aicunet import AICUNet
from model.a_unet import A_UNet
from model.m_unet import M_UNet
from model.rdam_unet import RDAM_UNet
from model.vm_unet import VMUNet
import torch.nn.functional as F
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# cam
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import argparse


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)["out"]
    
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

def main(args):
    # 加载图像
    img_path = "/mnt/e/VScode/WS-Hub/WS-U2net/U-2-Net/Image1 - 003.jpeg"
    img = Image.open(img_path).resize((224,224)).convert('RGB')

    # 预处理图像
    rgb_img = np.float32(img) / 255

    # 将图片转为tensor
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # 加载模型权重
    weights_path = args.weight_path
    checkpoint = torch.load(weights_path)
    state_dict = checkpoint['model']

    # 加载模型
    model_map = {
            "u2net_full"                    : u2net_full_config(),
            "u2net_lite"                    : u2net_lite_config(),
            "unet"                          : UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "ResD_unet"                     : ResD_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "a_unet"                        : A_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "m_unet"                        : M_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "rdam_unet"                     : RDAM_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "aicunet"                       : AICUNet(in_channels=3, n_classes=4, base_channels=32, p=args.dropout_p),
            "vm_unet"                       : VMUNet(input_channels=3, num_classes=4),
            "Segnet"                        : SegNet(n_classes=4, dropout_p=args.dropout_p),
            "pspnet"                        : PSPNet(classes=4, dropout=args.dropout_p, pretrained=False),
            "deeplabv3_resnet50"            : deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=4),
            "deeplabv3_resnet101"           : deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=4),
            "deeplabv3_mobilenetv3_large"   : deeplabv3_mobilenetv3_large(aux=False, pretrain_backbone=False, num_classes=4)
        }
    model = model_map.get(args.model)
    if not model:
        raise ValueError(f"Invalid model name: {args.model}")

    # 加载模型权重
    model.load_state_dict(state_dict)

    # 确保在推理模式下运行模型
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
    output = model(input_tensor)
    if isinstance(output, dict):
        output = output['out']

    # 模型输出为字典
    # model = SegmentationModelOutputWrapper(model)
    # output = model(input_tensor)
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

    # 分割类别
    sem_classes = [
        '__background__', 'Organic matter', 'Organic pore', 'Inorganic pore'
    ]

    # 获取类索引
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    OM_category = sem_class_to_idx["Organic matter"]
    OP_category = sem_class_to_idx["Organic pore"]
    IP_category = sem_class_to_idx["Inorganic pore"]

    # OM
    OM_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    OM_mask_uint8 = 255 * np.uint8(OM_mask == OM_category)
    OM_mask_float = np.float32(OM_mask == OM_category)

    # OP
    OP_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    OP_mask_uint8 = 255 * np.uint8(OP_mask == OP_category)
    OP_mask_float = np.float32(OP_mask == OP_category)

    # IP
    IP_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    IP_mask_uint8 = 255 * np.uint8(IP_mask == IP_category)
    IP_mask_float = np.float32(IP_mask == IP_category)

    # 指定目标层
    # target_layers = [model.up4.conv]

    # target_layers = [model.upsample_5]

    target_layers = [model.out_conv]

    # OM
    OM_targets = [SemanticSegmentationTarget(OM_category, OM_mask_float)]
    with GradCAM(model=model,
                target_layers=target_layers,
                ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=OM_targets)[0, :]
        OM_cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # OP
    OP_targets = [SemanticSegmentationTarget(OP_category, OP_mask_float)]
    with GradCAM(model=model,
                target_layers=target_layers,
                ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=OP_targets)[0, :]
        OP_cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # IP
    IP_targets = [SemanticSegmentationTarget(IP_category, IP_mask_float)]
    with GradCAM(model=model,
                target_layers=target_layers,
                ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=IP_targets)[0, :]
        IP_cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    name = args.name    
    # 保存图片
    save_path = "/mnt/e/VScode/WS-Hub/WS-U2net/U-2-Net/results/cam_img"
    OM_cam_img = Image.fromarray(OM_cam_image)
    OM_cam_img.save(os.path.join(save_path, f"{name}_OM.jpg"))

    OP_cam_img = Image.fromarray(OP_cam_image)
    OP_cam_img.save(os.path.join(save_path, f"{name}_OP.jpg"))

    IP_cam_img = Image.fromarray(IP_cam_image)
    IP_cam_img.save(os.path.join(save_path, f"{name}_IOP.jpg"))

    print("映射完成！！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cam images")

    parser.add_argument('--weight_path',        type=str,
                        default="/mnt/e/VScode/WS-Hub/WS-U2net/U-2-Net/results/save_weights/msaf_unetv2/L: DiceLoss--S: CosineAnnealingLR/optim: AdamW-lr: 0.0008-wd: 1e-06/2025-03-10_09-44-44/model_best_ep_121.pth")
    
    parser.add_argument('--model',              type=str, 
                        default="msaf_unetv2", 
                                                help=" unet, ResD_unet, msaf_unet, msaf_unetv2, a_unet, a_unetv2, m_unet, aicunet\
                                                Segnet, deeplabv3_resnet50, deeplabv3_mobilenetv3_large, pspnet, u2net_full, u2net_lite,")
    
    parser.add_argument('--name',               type=str,
                        default='msaf_unetv2',  help="names of saving images")
    args = parser.parse_args()
    main(args)