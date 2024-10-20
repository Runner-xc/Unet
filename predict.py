import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from SEM_Data import SEM_DATA
import argparse
import time
from model.u2net import u2net_full_config, u2net_lite_config
from model.unet import UNet
from model.DL_unet import DL_UNet
from model.SED_unet import SED_UNet
from tqdm import tqdm
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from loss_fn import *
from metrics import Evaluate_Metric
import utils.transforms as T

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
    
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据集
    test_dataset = SEM_DATA(args.data_path, transforms=SODPresetEval([320, 320]))
    
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
    elif args.model_name == 'DL_unet':
        model = DL_UNet(
                    in_channels=3,
                    n_classes=4,
                    p=0)
    elif args.model_name == 'SED_unet':
        model = SED_UNet(
                    in_channels=3,
                    n_classes=4,
                    p=0,
                    base_channels=32)
    else:
        raise ValueError(f"model name error")
    
    
    # 加载模型权重
    pretrain_weights = torch.load(args.weights_path)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
        
    model = model.to(device)
     
    if args.single:
        # test 单张
        path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/Image1 - 003.jpeg'
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Resize((1000,1000))(img)
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
        if not os.path.exists("predict/"):
            os.mkdir("predict/")
        # pred_img_pil.save(f"predict/SED_unet_Dice_cos_adamw_lr:8e-4_wd:1e-6.png")
        pred_img_pil.save(f"predict/test.png")
        print("预测完成!")
        
    else:
        # test 多张
        model.eval()
        test_loader = tqdm(test_loader, desc=f"  Validating  😀", leave=False)
        count = 0
        save_path = f"{args.save_path}/{args.model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for data in test_loader:
            images, masks = data[0].to(device), data[1].to(device)
            logits = model(images)  # [1, 4, 320, 320]
            
            # 使用 argmax 获取每个像素点预测最大的类别索引
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
            pred_img_pil.save(f"{save_path}/pred_{count}.png")
            count += 1
        
        print("预测完成！")
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/test_rock_sem_224.csv')
    parser.add_argument('--base_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='SED_unet', help='model name must be unet, u2net_full or u2net_lite or DL_unet or SED_unet')
    parser.add_argument('--weights_path', type=str, 
                        default='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/results/save_weights/SED_unet/L: DiceLoss--S: CosineAnnealingLR/optim: AdamW-lr: 0.0008-wd: 1e-06/2024-10-20_10:17:37/model_best.pth')
    parser.add_argument('--save_path', type=str, default='/mnt/c/VScode/WS-Hub/WS-U2net/results/predict/')
    parser.add_argument('--single', type=bool, default=True, help='test one img or not')
    
    args = parser.parse_args()
    main(args)
