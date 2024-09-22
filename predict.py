import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from SEM_Data import SEM_DATA
import argparse
import time
from model.u2net import u2net_full_config, u2net_lite_config
from model.unet import UNet
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
                    n_classes=4)
    elif args.model_name == 'u2net_full':
        model = u2net_full_config()
    elif args.model_name == 'u2net_lite':
        model = u2net_lite_config()
    else:
        raise ValueError(f"model name must be unet, u2net_full or u2net_lite")
    
    
    # 加载模型权重
    pretrain_weights = torch.load(args.weights_path, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
        
    model = model.to(device)
    
    # # 预测
    # loss_fn = CrossEntropyLoss()  # DiceLoss()  Focal_Loss()
    # Metrics = Evaluate_Metric()

    # val_mean_loss, Metric_list = evaluate(model, device, test_loader, loss_fn, Metrics)
    # val_mean_loss = val_mean_loss / len(test_loader)

    # # 评价指标 metrics = [recall, precision, dice, f1_score]
    # val_metrics ={}
    # val_metrics["Recall"] = Metric_list[0]
    # val_metrics["Precision"] = Metric_list[1]
    # val_metrics["Dice"] = Metric_list[2]
    # val_metrics["F1_scores"] = Metric_list[3]
            
    # # 保存指标
    # metrics_table_header = ['Metrics_Name', 'Mean']
    # metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores']
    # time_s = f" 👉 time :{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} 👈\n"
    # metrics_dict = {scores : val_metrics[scores] for scores in metrics_table_left}
    # metrics_table = [[metric_name,
    #                     metrics_dict[metric_name][3],
    #                 ]
    #                     for metric_name in metrics_table_left
    #                 ]
    # table_s = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
    # loss_s = F" 👉 mean_loss :{val_mean_loss:.3f} 👈\n"

    # # 记录每个epoch对应的train_loss、lr以及验证集各指标
    # write_info = time_s + '\n' + table_s + '\n' + '\n' + loss_s + '\n' + '\n'

    # # 打印结果
    # print(write_info)
    
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
        pred_img_pil.save(f"predict1.png")
        
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
        
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/test_data.csv')
    parser.add_argument('--base_size', type=int, default=320)
    parser.add_argument('--model_name', type=str, default='unet', help='model name must be unet, u2net_full or u2net_lite')
    parser.add_argument('--weights_path', type=str, default='/mnt/c/VScode/WS-Hub/WS-U2net/results/save_weights/unet/model_ep:58.pth')
    parser.add_argument('--save_path', type=str, default='/mnt/c/VScode/WS-Hub/WS-U2net/results/predict/')
    parser.add_argument('--single', type=bool, default=True, help='test one img or not')
    
    args = parser.parse_args()
    main(args)
