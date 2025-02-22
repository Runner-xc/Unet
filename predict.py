import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from SEM_Data import SEM_DATA
import argparse
import time
from model.deeplabv3_model import deeplabv3_resnet50
from model.pspnet import PSPNet
from model.Segnet import SegNet
from model.u2net import u2net_full_config, u2net_lite_config
from model.unet import *
from tqdm import tqdm
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from loss_fn import *
from metrics import Evaluate_Metric
import utils.transforms as T
from typing import Union, List

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

t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
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
    if args.model_name == 'unet':
        model = UNet(
                    in_channels=3,
                    n_classes=4,
                    p=0)
    elif args.model_name == 'u2net_full':
        model = u2net_full_config()
    elif args.model_name == 'u2net_lite':
        model = u2net_lite_config()

    elif args.model_name == 'Segnet':
        model = SegNet(n_classes=4, dropout_p=0)
    
    elif args.model_name == "pspnet":
        model = PSPNet(num_classes=4, dropout_p=0, use_aux=False)

    elif args.model_name == "deeplabv3":
        model = deeplabv3_resnet50(num_classes=4, pretrain_backbone=False, aux=False)
    
    elif args.model_name == "ResD_unet":
        model = ResD_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=0)
    elif args.model_name == "a_unet":
        model = A_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=0)
    elif args.model_name == "m_unet":
        model = M_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=0)
    elif args.model_name == "msaf_unet":
        model = MSAF_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=0)

    else:
        raise ValueError(f"model name error")
    
    
    # 加载模型权重
    pretrain_weights = torch.load(args.weights_path)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
        
    model = model.to(device)
    
    Metric = Evaluate_Metric()  
    if args.single:
        # test 单张
        path = '/mnt/e/VScode/WS-Hub/WS-U2net/U-2-Net/Image1 - 003.jpeg'
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        
        img = torchvision.transforms.ToTensor()(img)
        # img = torchvision.transforms.Resize((1024,1024))(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        logits = model(img)
        pred_mask = torch.argmax(logits, dim=1)  # [1, 320, 320]
        pred_mask = pred_mask.squeeze(0)  # [320, 320]        
        pred_mask = pred_mask.to(torch.uint8)       
        pred_mask = pred_mask.cpu()
        pred_mask_np = pred_mask.numpy()
        pred_img_pil = Image.fromarray(pred_mask_np)
        # 保存图片
        if not os.path.exists("single_predict/"):
            os.mkdir("single_predict/")
        pred_img_pil.save(f"single_predict/unet_1.png")        
        print("预测完成!")
       
    else:
        # test 多张
        with torch.no_grad():
            model.eval()
            Metric_list = np.zeros((6, 4))
            test_loader = tqdm(test_loader, desc=f"  Validating  😀", leave=False)
            save_path = f"{args.save_path}/{args.model_name}"
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
                
                # 移除批次维度，如果存在
                pred_mask = pred_mask.squeeze(0)  # [320, 320]
                
                # 确保 pred_mask 是 uint8 类型
                pred_mask = pred_mask.to(torch.uint8)
                
                # 将 pred_mask 转移到 CPU 上
                pred_mask = pred_mask.cpu()
                
                # 将 torch.Tensor 转换为 numpy 数组
                pred_mask_np = pred_mask.numpy()
                
                # 将 numpy 数组转换为 PIL 图像
                pred_img_pil = Image.fromarray(pred_mask_np)
                
                # 保存图片
                if not os.path.exists(f"{save_path}/pred_img/"):
                    os.makedirs(f"{save_path}/pred_img/")
                pred_img_pil.save(f"{save_path}/pred_img/{os.path.splitext(img_name)[0]}.png")
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
        write_info = f"{args.model_name}" + "\n" + table_s
        file_path = f'{save_path}/scores/{t}.txt'
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path)) 
        with open(file_path, "a") as f:
                    f.write(write_info)
        print(table_s)
        print("预测完成！")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/Unet/Datasets/CSV/test_rock_sem_chged_256_a50_c80.csv')
    parser.add_argument('--base_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='msaf_unet', help=' unet, m_unet, a_unet, ResD_unet, msaf_unet, u2net_full, u2net_lite, Segnet, pspnet, deeplabv3,')
    parser.add_argument('--weights_path', type=str, 
                        default='/root/Unet/results/save_weights/msaf_unet/L: DiceLoss--S: CosineAnnealingLR/optim: AdamW-lr: 0.0008-wd: 1e-06/2025-02-21_17:47:19/model_best_ep:38.pth')
    parser.add_argument('--save_path', type=str, default='/root/Unet/results/predict')
    parser.add_argument('--single', type=bool, default=False, help='test single img or not')
    
    args = parser.parse_args()
    main(args)
