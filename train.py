import torch
import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from SEM_Data import SEM_DATA
from utils import data_split
import argparse
import os
from torch.optim import Adam, SGD, RMSprop, AdamW
import time
from model.u2net import u2net_full_config, u2net_lite_config
from tqdm import tqdm
from tabulate import tabulate
from utils.train_and_eval import *
from loss_fn import *
from torch.amp import GradScaler, autocast
from metrics import Evaluate_Metric
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

now = time.localtime()
detailed_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", now)
log_name = f"training_log_{detailed_time_str}.txt"

def main(args):
    initial_time = time.time()

    """参数列表"""
    params = vars(args)
    params_dict = {}
    params_dict['Parameter'] = [str(p[0]) for p in list(params.items())]
    params_dict['Value'] = [str(p[1]) for p in list(params.items())]
    params_header = ['Parameter', 'Value']
    """打印参数"""
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    

    # 定义设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    save_root_path = args.save_root_path
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    # 日志保存路径
    writer = SummaryWriter(f'./results/logs/{args.log_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')

    """——————————————————————————————————————————————加载数据集——————————————————————————————————————————————"""
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    # 预处理
    img_compose  =  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.GaussianBlur((3, 3)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Normalize(mean=[0.485], std=[0.229])])
    
    mask_compose =  transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)])
    
    train_datasets, val_datasets, train_datasets = data_split.data_split_to_train_val_test(args.data_path, train_ratio, val_ratio,
                                                                                          save_path='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV')# 保存划分好的数据集路径
    # 读取数据集
    train_datasets = SEM_DATA(train_datasets, 
                              img_transforms=img_compose, 
                              mask_transforms=mask_compose)
    
    val_datasets = SEM_DATA(val_datasets, 
                            img_transforms=img_compose, 
                            mask_transforms=mask_compose)
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataloader = DataLoader(train_datasets, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    val_dataloader = DataLoader(val_datasets, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    """——————————————————————————————————————————————模型 配置———————————————————————————————————————————————"""   
    # 加载模型
    if args.model =="u2net_full":
        model = u2net_full_config()
    elif args.model =="u2net_lite":
        model = u2net_lite_config()
    else:
        raise ValueError("model must be 'u2net_full' or 'u2net_lite'.")
    model.to(device)

    # 优化器
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) # 会出现梯度爆炸或消失

    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

    elif args.optimizer == 'RMSprop':

        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8)
    else:
        raise ValueError("optimizer must be 'AdamW', 'SGD' or 'RMSprop'.")
    
    # 损失函数
    if args.loss_fn == 'CrossEntropyLoss':
        loss_fn = CrossEntropyLoss()
    elif args.loss_fn == 'DiceLoss':
        loss_fn = DiceLoss()
    elif args.loss_fn == 'SurfaceLoss':
        loss_fn = SurfaceLoss()
    elif args.loss_fn == 'FocalLoss':
        loss_fn = Focal_Loss()
    else:
        raise ValueError("loss function must be 'CrossEntropy', 'FocalLoss', 'DiceLoss' or 'SurfaceLoss'.")
    # 缩放器
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    Metrics = Evaluate_Metric()
    
    """——————————————————————————————————————————————训练 验证——————————————————————————————————————————————"""
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch


  
    current_mae, current_f1 = float('inf'), 0.0
    
    for epoch in range(start_epoch, end_epoch):
        print(f"--Training-- 👉 epoch {epoch+1}/{end_epoch}😀😀😀")
        # 记录时间
        start_time = time.time()
        # 训练
        train_OM_loss, train_OP_loss, train_IOP_loss, train_mean_loss = train_one_epoch(model, 
                                                                                        optimizer, 
                                                                                        epoch, 
                                                                                        train_dataloader, 
                                                                                        device=device, 
                                                                                        loss_fn=loss_fn, 
                                                                                        scaler=scaler) # loss
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "Metrics": Metrics.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # 求平均
        train_OM_loss = train_OM_loss / len(train_dataloader)
        train_OP_loss = train_OP_loss / len(train_dataloader)
        train_IOP_loss = train_IOP_loss / len(train_dataloader)
        train_mean_loss = train_mean_loss / len(train_dataloader)

        # 记录日志
        writer.add_scalars('train/Loss', 
                           {'Mean':train_mean_loss, 
                             'OM': train_OM_loss, 
                             'OP': train_OP_loss, 
                             'IOP': train_IOP_loss},
                           epoch)

        # 结束时间
        end_time = time.time()
        train_cost_time = end_time - start_time

        # 打印
        print(f"[epoch: {epoch}]\n"
              f"train_OM_loss: {train_OM_loss:.3f}\n"
              f"train_OP_loss: {train_OP_loss:.3f}\n"
              f"train_IOP_loss: {train_IOP_loss:.3f}\n"
              f"train_mean_loss: {train_mean_loss:.3f}\n"
              f"train_cost_time: {train_cost_time:.2f}s")
        
        # 验证
        if epoch % args.eval_interval == 0 or epoch == end_epoch - 1:

            print(f"--Validation-- 👉 epoch {epoch+1}/{end_epoch}😀😀😀")
            # 记录验证开始时间
            start_time = time.time()
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            val_OM_loss, val_OP_loss, val_IOP_loss, val_mean_loss, Metric_list= evaluate(model, device, val_dataloader, loss_fn, Metrics) # val_loss, recall, precision, f1_scores
            
            # 求平均
            val_OM_loss = val_OM_loss / len(val_dataloader)
            val_OP_loss = val_OP_loss / len(val_dataloader)
            val_IOP_loss = val_IOP_loss / len(val_dataloader)
            val_mean_loss = val_mean_loss / len(val_dataloader)

            # 评价指标 metrics = [recall, precision, dice, f1_score]
            val_metrics ={}
            val_metrics[epoch] = epoch
            val_metrics["Recall"] = Metric_list[0]
            val_metrics["Precision"] = Metric_list[1]
            val_metrics["Dice"] = Metric_list[2]
            val_metrics["F1_scores"] = Metric_list[3]

            # 验证====结束时间
            end_time = time.time()
            val_cost_time = end_time - start_time

            # 打印结果
            print(f"[epoch: {epoch}]\n"
                  f"val_OM_loss: {val_OM_loss:.3f}\n"
                  f"val_OP_loss: {val_OP_loss:.3f}\n"
                  f"val_IOP_loss: {val_IOP_loss:.3f}\n"
                  f"val_mean_loss: {val_mean_loss:.3f}\n"
                  f"val_cost_time: {val_cost_time:.2f}s")
            
            # 记录日志
            tb = args.tb
            if tb:
                writer.add_scalars('val/Loss', 
                                {'Mean':val_mean_loss, 
                                    'OM': val_OM_loss, 
                                    'OP': val_OP_loss, 
                                    'IOP': val_IOP_loss},
                                epoch)
                
                writer.add_scalars('val/Dice',
                                {'Mean':val_metrics['Dice'][0],
                                    'OM': val_metrics['Dice'][1],
                                    'OP': val_metrics['Dice'][2],
                                    'IOP': val_metrics['Dice'][3]},
                                epoch)
                
                writer.add_scalars('val/Precision', 
                                {'Mean':val_metrics['Precision'][0],
                                    'OM': val_metrics['Precision'][1], 
                                    'OP': val_metrics['Precision'][2], 
                                    'IOP': val_metrics['Precision'][3]},
                                epoch)
                
                writer.add_scalars('val/Recall', 
                                {'Mean':val_metrics['Recall'][0], 
                                    'OM': val_metrics['Recall'][1], 
                                    'OP': val_metrics['Recall'][2], 
                                    'IOP': val_metrics['Recall'][3]},
                                epoch)
                writer.add_scalars('val/F1', 
                                {'Mean':val_metrics['F1_scores'][0], 
                                    'OM': val_metrics['F1_scores'][1], 
                                    'OP': val_metrics['F1_scores'][2], 
                                    'IOP': val_metrics['F1_scores'][3]},
                                epoch) 
                # writer.add_scalars('val/F2', 
                #                 {'Mean':val_metrics['F2_scores'][0], 
                #                     'OM': val_metrics['F2_scores'][1], 
                #                     'OP': val_metrics['F2_scores'][2], 
                #                     'IOP': val_metrics['F2_scores'][3]},
                #                 epoch)  

                # writer.add_scalars('val/Jaccard_index',
                #                 {'Mean':val_metrics['Jaccard_scores'][0],
                #                     'OM': val_metrics['Jaccard_scores'][1],
                #                     'OP': val_metrics['Jaccard_scores'][2],
                #                     'IOP': val_metrics['Jaccard_scores'][3]},
                #                 epoch)   

                # writer.add_scalars('val/Accuracy',
                #                 {'Mean':val_metrics['Accuracy_scores'][0],
                #                     'OM': val_metrics['Accuracy_scores'][1],
                #                     'OP': val_metrics['Accuracy_scores'][2],
                #                     'IOP': val_metrics['Accuracy_scores'][3]},
                #                 epoch)
            
            
            # 保存指标
            metrics_table_header = ['Metrics_Name', 'mean', 'OM', 'OP', 'IOP']
            metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores']
            epoch_s = f" 👉 epoch :{epoch} 👈\n"
            time_s = f" 👉 time :{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} 👈\n"
            cost_s = f" 👉 cost_time :{val_cost_time/60:.2f}mins 👈\n"
            metrics_dict = {scores : val_metrics[scores] for scores in metrics_table_left}
            metrics_table = [[metric_name,
                              metrics_dict[metric_name][0],
                              metrics_dict[metric_name][1],
                              metrics_dict[metric_name][2],
                              metrics_dict[metric_name][3]]
                             for metric_name in metrics_table_left
                            ]
            table_s = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
            loss_s = F" 👉 mean_loss :{val_mean_loss:.3f} 👈\n"

            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            write_info = epoch_s + time_s + '\n' + table_s + '\n' + '\n' + loss_s + cost_s + '\n' + '\n'

            # 打印结果
            print(write_info)

            # 保存结果
            results_file = f"{log_name}.txt"
            file_path = os.path.join(save_root_path, results_file)
            with open(file_path, "a") as f:
                f.write(write_info)

        # 保存best模型
        if not os.path.exists("results/save_weights"):
            os.makedirs("results/save_weights")

        if current_mae >= val_mean_loss and current_f1 <= val_metrics["F1_scores"][3]:
            torch.save(save_file, "results/save_weights/model_best.pth")

        # only save latest 10 epoch weights
        if os.path.exists(f"results/save_weights/model_ep:{epoch-10}.pth"):
            os.remove(f"results/save_weights/model_ep:{epoch-10}.pth")

        torch.save(save_file, f"results/save_weights/model_ep:{epoch}.pth")

    total_time = time.time() - initial_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("====training over. total time: {}".format(total_time_str))
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train model on SEM stone dataset")

    parser.add_argument('--data_path', type=str, default="U-2-Net/SEM_DATA/CSV/SEM_int_path.csv", help="path to csv dataset")
    
    parser.add_argument('--model', type=str, default="u2net_full", help="model name")
    parser.add_argument('--loss_fn', type=str, default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--save_root_path', type=str, default='results/save_scores', help="root path to save scores on training and valing")
    parser.add_argument('--log_name', type=str, default='u2net')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', type=bool, default=True, help='use mixed precision training or not')
    parser.add_argument('--tb', type=bool, default=False, help='use tensorboard or not')

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--end_epoch', type=int, default=100, help='ending epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--eval_interval', type=int, default=10, help='interval for evaluation')
  

    args = parser.parse_args()
    main(args)