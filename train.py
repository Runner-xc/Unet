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
from model.unet import UNet
from tqdm import tqdm
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from utils import param_modification
from utils import write_experiment_log
from loss_fn import *
from torch.amp import GradScaler, autocast
from metrics import Evaluate_Metric
from torch.utils.tensorboard import SummaryWriter
import utils.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data

class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data
    

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


detailed_time_str = time.strftime("%Y-%m-%d")

def main(args):

    """——————————————————————————————————————————————打印初始配置———————————————————————————————————————————————"""
    
    params = vars(args)
    """映射参数序号到参数名称"""
    param_map = {
        2: 'lr',
        3: 'l1_lambda',
        4: 'l2_lambda',
        5: 'dropout_p',
        6: 'eval_interval',
        7: 'batch_size',
        8: 'optimizer',
        9: 'small_data',
        10: 'Tmax',
        11: 'eta_min',
        12: 'last_epoch',
        13: 'save_weights',
        14: 'scheduler',
        15: 'model',
        16: 'loss_fn',
        17: 'split_flag'
    }

    """筛选需要打印的参数"""
    printed_params = list(param_map.values())
    params_dict = {}
    params_dict['Parameter'] = printed_params
    params_dict['Value'] = [str(params[p]) for p in printed_params]
    params_header = ['Parameter', 'Value']

    """打印参数"""
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    
    """——————————————————————————————————————————————记录修改配置———————————————————————————————————————————————"""
    initial_time = time.time()
    x = input("是否需要修改配置参数：\n 1. 不修改, 继续。 \n 2. lr \n 3. l1_lambda \n 4. l2_lambda \n 5. dropout_p \n \
6. eval_interval \n 7. batch_size \n 8. optimizer \n 9. small_data \n 10. Tmax \n 11. eta_min \n \
12. last_epoch \n 13. save_weights \n 14. scheduler \n 15. model \n 16. loss_fn \n 17. split_flag \n\
请输入需要修改的参数序号（int）： ")
    
    args, contents = param_modification.param_modification(args, x)
    save_modification_path = f"/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/results/modification_log/{args.model}/{detailed_time_str}/lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}.md"
    if not os.path.exists(os.path.dirname(save_modification_path)):
        os.makedirs(os.path.dirname(save_modification_path))
    write_experiment_log.write_exp_logs(save_modification_path, contents) 
        
    """——————————————————————————————————————————————模型 配置———————————————————————————————————————————————"""  
    
    # 定义设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    save_scores_path = args.save_scores_path
    if not os.path.exists(save_scores_path):
        os.makedirs(save_scores_path)
         
    # 加载模型
    
    assert args.model in ["u2net_full", "u2net_lite", "unet"], \
        f"model must be 'u2net_full' or 'u2net_lite' or 'unet', but got {args.model}"
    if args.model =="u2net_full":
        model = u2net_full_config()
    elif args.model =="u2net_lite":
        model = u2net_lite_config()
    elif args.model == "unet":
        if args.small_data:
            
            # 重新设定dropout rate
            setattr(args, 'dropout_p', 0.5)
            model = UNet(
                         in_channels=3, n_classes=4, base_channels=64, bilinear=True, p=args.dropout_p)
        else:
            model = UNet(
                         in_channels=3, n_classes=4, base_channels=64, bilinear=True, p=args.dropout_p)
 
    else:
        if args.small_data:
            
            # 重新设定dropout rate
            setattr(args, 'dropout_p', 0.5)
            model = UNet(
                         in_channels=3, n_classes=4, base_channels=64, bilinear=True, p=args.dropout_p)
        else:
            model = UNet(
                         in_channels=3, n_classes=4, base_channels=64, bilinear=True, p=args.dropout_p)
    
    # 初始化模型
    kaiming_initial(model)
    model.to(device)  

    # 优化器
    
    assert args.optimizer in ['AdamW', 'SGD', 'RMSprop'], \
        f'optimizer must be AdamW, SGD, RMSprop but got {args.optimizer}'
        
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                        #   weight_decay=args.wd
                          ) # 会出现梯度爆炸或消失

    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                        # weight_decay=args.wd
                        )

    elif args.optimizer == 'RMSprop':

        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, 
                            # weight_decay=args.wd
                            )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                        #   weight_decay=args.wd
                          )
    
    # 调度器
    """
        "LambdaLR",
        "MultiplicativeLR",
        "StepLR",
        "MultiStepLR",
        "ConstantLR",
        "LinearLR",
        "ExponentialLR",
        "SequentialLR",
        "CosineAnnealingLR",
        "ChainedScheduler",
        "ReduceLROnPlateau",
        "CyclicLR",
        "CosineAnnealingWarmRestarts",
        "OneCycleLR",
        "PolynomialLR",
        "LRScheduler",
    """
    assert args.scheduler in ['CosineAnnealingLR', 'ReduceLROnPlateau'], \
            f'scheduler must be CosineAnnealingLR 、ReduceLROnPlateau, but got {args.scheduler}'
    if args.scheduler == 'CosineAnnealingLR':
        if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0 and "initial_lr" not in optimizer.param_groups[0]:
            # 如果optimizer的param_groups中没有initial_lr，则添加initial_lr
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.Tmax, 
                                      eta_min=args.eta_min,
                                      verbose=True,
                                      last_epoch=args.last_epoch)
        
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='min', 
                                      factor=0.1, 
                                      patience=10, 
                                      threshold=1e-4, 
                                      threshold_mode='rel', 
                                      cooldown=0, 
                                      min_lr=0, 
                                      eps=1e-8)
    else:
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.Tmax, 
                                      eta_min=args.eta_min,
                                      verbose=True,
                                      last_epoch=args.last_epoch)
        
    # 损失函数
    
    assert args.loss_fn in ['CrossEntropyLoss', 'DiceLoss', 'FocalLoss']
    if args.loss_fn == 'CrossEntropyLoss':
        loss_fn = CrossEntropyLoss()
    elif args.loss_fn == 'DiceLoss':
        loss_fn = DiceLoss()
    elif args.loss_fn == 'FocalLoss':
        loss_fn = Focal_Loss()
    else:
        loss_fn = DiceLoss()
    
    # 缩放器
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    Metrics = Evaluate_Metric()
    
    # 日志保存路径
    save_logs_path = f"./results/logs/{args.model}"
    
    if not os.path.exists(save_logs_path):
        os.makedirs(save_logs_path)
    writer = SummaryWriter(f'{save_logs_path}/{detailed_time_str}/lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}')
    """——————————————————————————————————————————————断点 续传———————————————————————————————————————————————"""
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        Metrics = checkpoint['Metrics']
        scheduler = checkpoint['scheduler']
        best_mean_loss = checkpoint['best_mean_loss']
        args = checkpoint['args']    
        
    
    """——————————————————————————————————————————————参数 列表———————————————————————————————————————————————"""
    
    """参数列表"""
    params = vars(args)
    params_dict = {}
    params_dict['Parameter'] = [str(p[0]) for p in list(params.items())]
    params_dict['Value'] = [str(p[1]) for p in list(params.items())]
    params_header = ['Parameter', 'Value']
    """打印参数"""
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))

    """——————————————————————————————————————————————加载数据集——————————————————————————————————————————————"""
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    
    
    # 预处理
    # img_compose  =  transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Resize((320, 320)),
    #                 transforms.GaussianBlur((3, 3)),
    #                 transforms.RandomHorizontalFlip(p=0.5),
    #                 transforms.Normalize(mean=[0.485], std=[0.229])])
    
    # mask_compose =  transforms.Compose([
    #                 transforms.Resize((320, 320)),
    #                 transforms.RandomHorizontalFlip(p=0.5)])
    

    # 划分数据集
    if args.small_data is not None:
        train_datasets, val_datasets, test_datasets = data_split.small_data_split_to_train_val_test(args.data_path, 
                                                                                                    num_small_data=args.small_data, 
                                                                                                    # train_ratio=0.8, 
                                                                                                    # val_ratio=0.1, 
                            save_path='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV',
                            flag=args.split_flag) 
    
    else:
        train_datasets, val_datasets, test_datasets = data_split.data_split_to_train_val_test(args.data_path, train_ratio=train_ratio, val_ratio=val_ratio,
                            save_path='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV',   # 保存划分好的数据集路径
                            flag=args.split_flag)

    # 读取数据集
    train_datasets = SEM_DATA(train_datasets, 
                            transforms=SODPresetTrain((320, 320), crop_size=320))
    
    val_datasets = SEM_DATA(val_datasets, 
                            transforms=SODPresetEval((320, 320)))
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataloader = DataLoader(train_datasets, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    val_dataloader = DataLoader(val_datasets, 
                                batch_size=4, 
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    """——————————————————————————————————————————————训练 验证——————————————————————————————————————————————"""
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch


  
    best_mean_loss, current_miou = float('inf'), 0.0
    current_mean_loss = float('inf')
    
    for epoch in range(start_epoch, end_epoch):
        
        print(f"✈✈✈✈✈ epoch : {epoch + 1} / {end_epoch} ✈✈✈✈✈✈")
        print(f"--Training-- 😀")
        # 记录时间
        start_time = time.time()
        # 训练
        total_loss = train_one_epoch(model, 
                                    optimizer, 
                                    epoch, 
                                    train_dataloader, 
                                    device=device, 
                                    loss_fn=loss_fn, 
                                    scaler=scaler,
                                    elnloss=args.elnloss,     #  Elastic Net正则化
                                    l1_lambda=args.l1_lambda,
                                    l2_lambda=args.l2_lambda) # loss

        
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "Metrics": Metrics.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # 求平均
        # train_OM_loss = OM_loss / len(train_dataloader)
        # train_OP_loss = OP_loss / len(train_dataloader)
        # train_IOP_loss = IOP_loss / len(train_dataloader)
        train_mean_loss = total_loss / len(train_dataloader)

        # 记录日志
        writer.add_scalars('train/Loss', 
                           {'Mean':train_mean_loss, 
                            #  'OM': train_OM_loss, 
                            #  'OP': train_OP_loss, 
                            #  'IOP': train_IOP_loss
                            },
                           epoch)

        # 结束时间
        end_time = time.time()
        train_cost_time = end_time - start_time

        # 打印

        print(
            #   f"train_OM_loss: {train_OM_loss:.3f}\n"
            #   f"train_OP_loss: {train_OP_loss:.3f}\n"
            #   f"train_IOP_loss: {train_IOP_loss:.3f}\n"
              f"train_mean_loss: {train_mean_loss:.3f}\n"
              f"train_cost_time: {train_cost_time:.2f}s\n")
        
        # 验证
        if epoch % args.eval_interval == 0 or epoch == end_epoch - 1:

            print(f"--Validation-- 😀")
            # 记录验证开始时间
            start_time = time.time()
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            val_mean_loss, Metric_list = evaluate(model, device, val_dataloader, loss_fn, Metrics) # val_loss, recall, precision, f1_scores

            # 求平均
            # val_OM_loss = val_OM_loss / len(val_dataloader)
            # val_OP_loss = val_OP_loss / len(val_dataloader)
            # val_IOP_loss = val_IOP_loss / len(val_dataloader)
            val_mean_loss = val_mean_loss / len(val_dataloader)
            
            
            # 更新调度器
            scheduler.step()

            # 评价指标 metrics = [recall, precision, dice, f1_score]
            val_metrics ={}
            val_metrics[epoch] = epoch
            val_metrics["Recall"] = Metric_list[0]
            val_metrics["Precision"] = Metric_list[1]
            val_metrics["Dice"] = Metric_list[2]
            val_metrics["F1_scores"] = Metric_list[3]
            val_metrics["mIoU"] = Metric_list[4]
            # 验证====结束时间
            end_time = time.time()
            val_cost_time = end_time - start_time

            # 打印结果
            print(
                #   f"val_OM_loss: {val_OM_loss:.3f}\n"
                #   f"val_OP_loss: {val_OP_loss:.3f}\n"
                #   f"val_IOP_loss: {val_IOP_loss:.3f}\n"
                  f"val_mean_loss: {val_mean_loss:.3f}\n"
                  f"val_cost_time: {val_cost_time:.2f}s\n\n")
            
            # 记录日志
            tb = args.tb
            if tb:
                writer.add_scalars('val/Loss', 
                                {'Mean':val_mean_loss},
                                epoch)
                
                writer.add_scalars('val/Dice',
                                {'Mean':val_metrics['Dice'][3],
                                },
                                epoch)
                
                writer.add_scalars('val/Precision', 
                                {'Mean':val_metrics['Precision'][3]},
                                epoch)
                
                writer.add_scalars('val/Recall', 
                                {'Mean':val_metrics['Recall'][3]},
                                epoch)
                writer.add_scalars('val/F1', 
                                {'Mean':val_metrics['F1_scores'][3]},
                                epoch) 
                
                writer.add_scalars('val/mIoU', 
                                {'Mean':val_metrics['mIoU']},
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
            metrics_table_header = ['Metrics_Name', 'Mean']
            metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU']
            epoch_s = f"✈✈✈✈✈ epoch : {epoch + 1} / {end_epoch} ✈✈✈✈✈✈\n"
            model_s = f"model : {args.model} \n"
            lr_s = f"lr : {args.lr} \n"
            wd_s = f"wd : {args.wd} \n"  #####
            l1_lambda = f"l1_lambda : {args.l1_lambda} \n"
            l2_lambda = f"l2_lambda : {args.l2_lambda} \n"
            scheduler_s = f"scheduler : {args.scheduler} \n"
            loss_fn_s = f"loss_fn : {args.loss_fn} \n"
            time_s = f"time : {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} \n"
            cost_s = f"cost_time :{val_cost_time / 60:.2f}mins \n"
            
            metrics_dict = {scores : val_metrics[scores] for scores in metrics_table_left}
            metrics_table = [[metric_name,
                              metrics_dict[metric_name][-1],
                            #   metrics_dict[metric_name][0],
                            #   metrics_dict[metric_name][1],
                            #   metrics_dict[metric_name][2]
                            ]
                             for metric_name in metrics_table_left
                            ]
            table_s = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
            loss_s = f"mean_loss : {val_mean_loss:.3f}  🍎🍎🍎\n"

            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            write_info = epoch_s + model_s + lr_s + l1_lambda + l2_lambda + loss_fn_s + scheduler_s + loss_s + table_s + '\n' + cost_s + time_s + '\n'

            # 打印结果
            print(write_info)

            # 保存结果
            results_file = f"{args.model}/{detailed_time_str}/lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}.txt"
            file_path = os.path.join(save_scores_path, results_file)
            with open(file_path, "a") as f:
                f.write(write_info)
        # loss清零           
               
       
        if args.save_weights:
            # 保存best模型
            save_weights_path = f"{args.save_weight_path}/{args.model}/lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}"  # 保存权重路径
            
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)

            if best_mean_loss >= val_mean_loss and current_miou <= val_metrics["mIoU"][-1]:
                torch.save(save_file, f"{save_weights_path}/model_best.pth")
                best_mean_loss = val_mean_loss
                current_miou = val_metrics["mIoU"][-1]

            # only save latest 10 epoch weights
            if os.path.exists(f"{save_weights_path}/model_ep:{epoch-10}.pth"):
                os.remove(f"{save_weights_path}/model_ep:{epoch-10}.pth")
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)
            torch.save(save_file, f"{save_weights_path}/model_ep:{epoch}.pth") 
        
        # 记录验证loss是否出现上升         
        count_bad_loss = 0
        if val_mean_loss <= current_mean_loss:
            current_mean_loss = val_mean_loss
                     
        elif val_mean_loss > current_mean_loss:
            count_bad_loss += 1 
    
        # 早停判断
        if count_bad_loss >= 5:
            print('验证loss异常，训练终止。。。')
            break

    total_time = time.time() - initial_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("====training over. total time: {}".format(total_time_str))
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train model on SEM stone dataset")
    
    # 保存路径
    parser.add_argument('--data_path', type=str, 
                        default="/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/rock_sem_224.csv", 
                        help="path to csv dataset")
    
    parser.add_argument('--save_scores_path', type=str, 
                        default='/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/results/save_scores', 
                        help="root path to save scores on training and valing")
    
    parser.add_argument('--save_weight_path', type=str,
                        default="/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/results/save_weights",
                        help="the path of save weights")
    # 模型配置
    parser.add_argument('--model', type=str, default="unet", 
                        help="'u2net_full' or 'u2net_lite' or 'unet'")
    
    parser.add_argument('--loss_fn', type=str, default='FocalLoss', 
                        help="'CrossEntropyLoss', 'FocalLoss', 'DiceLoss'.")
    
    parser.add_argument('--optimizer', type=str, default='AdamW', 
                        help="'AdamW', 'SGD' or 'RMSprop'.")
    
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', 
                        help="'CosineAnnealingLR', 'ReduceLROnPlateau'.")
    
    # 正则化
    parser.add_argument('--elnloss', type=bool, default=True, help='use elnloss or not')
    parser.add_argument('--l1_lambda', type=float, default=0.001, help="L1 factor")
    parser.add_argument('--l2_lambda', type=float, default=0.001, help=' L2 factor')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='dropout rate')
    
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resume', type=str, default=None, help="the path of weight for resuming")
    parser.add_argument('--amp', type=bool, default=True, help='use mixed precision training or not')
    parser.add_argument('--tb', type=bool, default=False, help='use tensorboard or not')   
    parser.add_argument('--split_flag', type=bool, default=False, help='split data or not')
    
    parser.add_argument('--save_weights', type=bool, default=True, help='save weights or not')
    
    # 训练参数
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--end_epoch', type=int, default=150, help='ending epoch')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    
    parser.add_argument('--eval_interval', type=int, default=10, help='interval for evaluation')
    parser.add_argument('--small_data', type=int, default=None, help='number of small data')
    parser.add_argument('--Tmax', type=int, default=20, help='the numbers of half of T for CosineAnnealingLR')
    parser.add_argument('--eta_min', type=float, default=0.0001, help='minimum of lr for CosineAnnealingLR')
    parser.add_argument('--last_epoch', type=int, default=5, help='start epoch of lr decay for CosineAnnealingLR')

    args = parser.parse_args()
    main(args)