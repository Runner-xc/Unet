import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from utils.rock_data import SEM_DATA
import argparse
import time
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
from tqdm import tqdm
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from utils.loss_fn import *
from utils.metrics import Evaluate_Metric
import utils.transforms as T
from typing import Union, List
from utils.slide_predict import SlidingWindowPredictor

class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data
    
t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据集
    test_dataset = SEM_DATA(args.data_path, transforms=SODPresetEval([256, 256]))
    
    num_workers = 4
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=num_workers)
    
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
    pretrain_weights = torch.load(args.weights_path, weights_only=False)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
        
    model = model.to(device)
    model.eval()
    Metric = Evaluate_Metric()

    # 创建优化后的预测器
    predictor = SlidingWindowPredictor(
        model=model,
        device=device,
        window_size=1024,
        stride=512  # 重叠步长设置为窗口大小的一半
    )

    if args.single:
        # 单张预测
        image_path = "/mnt/e/VScode/WS-Hub/WS-U2net/U-2-Net/Image1 - 003.jpeg"
        # 滑窗预测
        if args.slide:
            save_path = f"{args.single_path}/{args.model}_sliding.png"
            predictor.predict(image_path, save_path)
        
        else:
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)
            img = torchvision.transforms.ToTensor()(img).to(device)
            # img = torchvision.transforms.Resize((1024,1024))(img)
            img = img.unsqueeze(0)
            logits = model(img)
            pred_mask = torch.argmax(logits, dim=1)  
            pred_mask = pred_mask.squeeze(0)          
            pred_mask = pred_mask.to(torch.uint8).cpu()       
            pred_mask_np = pred_mask.numpy()
            pred_img_pil = Image.fromarray(pred_mask_np)
            # 保存图片
            single_path = args.single_path
            if not os.path.exists(single_path):
                os.mkdir(single_path)
            pred_img_pil.save(f"{single_path}/{args.model}.png")        
            print("预测完成!")
       
    else:
        # test 多张
        with torch.no_grad():
            Metric_list = np.zeros((6, 4))
            test_loader = tqdm(test_loader, desc=f"  Validating  😀", leave=False)
            save_path = f"{args.save_path}/{args.model}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            for data in test_loader:
                images, masks = data[0][0].to(device), data[0][1].to(device)
                img_name = data[1][0]
                logits = model(images)  # [1, 4, 320, 320]
                masks = masks.to(torch.int64)
                masks = masks.squeeze(1)
                
                # 使用 argmax 获取每个像素点预测最大的类别索引
                pred_mask = torch.softmax(logits, dim=1)
                metrics = Metric.update(pred_mask, masks)
                Metric_list += metrics
                pred_mask = torch.argmax(logits, dim=1)  # [1, 320, 320]

                pred_mask = pred_mask.squeeze(0)  # [320, 320]
                pred_mask = pred_mask.to(torch.uint8)
                pred_mask = pred_mask.cpu()
                pred_mask_np = pred_mask.numpy()
                pred_img_pil = Image.fromarray(pred_mask_np)
                
                # 保存图片
                if not os.path.exists(f"{save_path}/pred_img/{t}"):
                    os.makedirs(f"{save_path}/pred_img/{t}")
                pred_img_pil.save(f"{save_path}/pred_img/{t}/{os.path.splitext(img_name)[0]}.png")
            Metric_list /= len(test_loader)
        metrics_table_header = ['Metrics_Name', 'Mean', 'OM', 'OP', 'IOP']
        metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']
        val_metrics = {}
        val_metrics["Recall"] = Metric_list[0]
        val_metrics["Precision"] = Metric_list[1]
        val_metrics["Dice"] = Metric_list[2]
        val_metrics["F1_scores"] = Metric_list[3]
        val_metrics["mIoU"] = Metric_list[4]
        val_metrics["Accuracy"] = Metric_list[5]
        metrics_dict = {scores : val_metrics[scores] for scores in metrics_table_left}
        metrics_table = [[metric_name,
                            metrics_dict[metric_name][-1],
                            metrics_dict[metric_name][0],
                            metrics_dict[metric_name][1],
                            metrics_dict[metric_name][2]
                        ]
                            for metric_name in metrics_table_left
                        ]
        table_s = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
        write_info = f"{args.model}" + "\n" + table_s
        file_path = f'{save_path}/scores/{t}.txt'
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path)) 
        with open(file_path, "a") as f:
                    f.write(write_info)
        print(table_s)
        print("预测完成！")         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str,       default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/datasets/CSV/test_shale_256.csv')
    parser.add_argument('--base_size',      type=int,       default=256 )
    parser.add_argument('--dropout_p',      type=int,       default=0   )
    parser.add_argument('--model',          type=str,       default='rdam_unet',     help='unet, a_unet, m_unet, rdam_unet, ResD_unet, Segnet, pspnet, deeplabv3, u2net_full, u2net_lite')
    parser.add_argument('--weights_path',   type=str,       
                                            default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/results/save_weights/rdam_unet/L_DiceLoss--S_CosineAnnealingLR/optim_AdamW-lr_0.0008-wd_0.0001/2025-04-01_17-40-23/model_best_ep_23.pth')
    
    parser.add_argument('--save_path',      type=str,       default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/results/predict')
    parser.add_argument('--single_path',    type=str,       default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/results/single_predict')
    parser.add_argument('--single',         type=bool,      default=False,          help='test single img or not')
    parser.add_argument('--slide',          type=bool,      default=False)
    
    args = parser.parse_args()
    main(args)
