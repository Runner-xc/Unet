import torch
import datetime
from torch.utils.data import DataLoader, Dataset
from SEM_Data import SEM_DATA
from utils import data_split
from utils.writing_logs import writing_logs
import argparse
import os
from torch.optim import Adam, SGD, RMSprop, AdamW
import time
from model.u2net import u2net_full_config, u2net_lite_config
from model.unet import *
from model.deeplabv3_model import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large
from model.pspnet import PSPNet
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
from typing import Union, List
from utils.run_tensorboard import run_tensorboard
from model.Segnet import SegNet


class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(base_size),
            # T.RandomCrop(crop_size),
            # T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data

class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Resize(base_size),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data
    

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


detailed_time_str = time.strftime("%Y-%m-%d_%H:%M:%S")

def main(args):

    """——————————————————————————————————————————————打印初始配置———————————————————————————————————————————————"""
    
    # 将args转换为字典
    params = vars(args)

    # 映射参数名称到参数
    param_map = {
        'lr'            : '1. lr',
        'wd'            : '2. wd',
        'l1_lambda'     : '3. l1_lambda',
        'l2_lambda'     : '4. l2_lambda',
        'elnloss'       : '5. elnloss',
        'dropout_p'     : '6. dropout_p',
        'model'         : '7. model',
        'loss_fn'       : '8. loss_fn',
        'optimizer'     : '9. optimizer',
        'scheduler'     : '10. scheduler',
        'Tmax'          : '11. Tmax',
        'eta_min'       : '12. eta_min',
        'last_epoch'    : '13. last_epoch',
        'save_flag'     : '14. save_flag',
        'batch_size'    : '15. batch_size',
        'small_data'    : '16. small_data',
        'eval_interval' : '17. eval_interval',
        'split_flag'    : '18. split_flag',
        'resume'        : '19. resume'
    }

    # 筛选需要打印的参数
    printed_params = list(param_map.keys())
    params_dict = {}
    params_dict['Parameter'] = [param_map[p] for p in printed_params]
    params_dict['Value'] = [str(params[p]) for p in printed_params if p in params]

    # 打印参数
    params_header = ['Parameter', 'Value']
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    
    """——————————————————————————————————————————————记录修改配置———————————————————————————————————————————————"""
    initial_time = time.time()
    if args.change_params:    
        x = input("是否需要修改配置参数：\n 0. 不修改, 继续。 \n\
请输入需要修改的参数序号（int）： ")
        
        args = param_modification.param_modification(args, x)
    save_modification_path = f"/root/projects/WS-U2net/U-2-Net/results/modification_log/{args.model}/L: {args.loss_fn}--S: {args.scheduler}"
            
    """——————————————————————————————————————————————模型 配置———————————————————————————————————————————————"""  
    
    # 定义设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

         
    # 加载模型
    
    assert args.model in ["u2net_full", "u2net_lite", "unet", "ResD_unet", "Segnet", "deeplabv3_resnet50", "deeplabv3_resnet101", "pspnet", "msaf_unet", "a_unet", "m_unet"], \
        f"wrong model: {args.model}"
    if args.model =="u2net_full":
        model = u2net_full_config()
    elif args.model =="u2net_lite":
        model = u2net_lite_config()
    # unet系列
    elif args.model == "unet":   
        model = UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "ResD_unet":
        model = ResD_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "a_unet":
        model = A_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    elif args.model == "m_unet":
        model = M_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)    
    elif args.model == "msaf_unet":
        model = MSAF_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p)
    # 其他模型        
    elif args.model == "Segnet":
        model = SegNet(n_classes=4, dropout_p=args.dropout_p)
    elif args.model == "pspnet":
        model = PSPNet(num_classes=4, use_aux=True, dropout_p=args.dropout_p)
    elif args.model == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=4)
    elif args.model == "deeplabv3_resnet101":
        model = deeplabv3_resnet101(aux=False, pretrain_backbone=False, num_classes=4)
    elif args.model == "deeplabv3_mobilenetv3_large":
        model = deeplabv3_mobilenetv3_large(aux=False, pretrain_backbone=False, num_classes=4)
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    
    # 初始化模型
    kaiming_initial(model)
    model.to(device)  

    # 优化器   TODO: 优化器参数
    
    assert args.optimizer in ['AdamW', 'SGD', 'RMSprop'], \
        f'optimizer must be AdamW, SGD, RMSprop but got {args.optimizer}'
        
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=args.wd
                          ) # 会出现梯度爆炸或消失

    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                        weight_decay=args.wd
                        )

    elif args.optimizer == 'RMSprop':

        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, 
                            weight_decay=args.wd
                            )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=args.wd
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
                                      patience=5, 
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
    
    assert args.loss_fn in ['CrossEntropyLoss', 'DiceLoss', 'FocalLoss', 'WDiceLoss', 'DWDLoss', 'IoULoss', 'dice_hd']
    if args.loss_fn == 'CrossEntropyLoss':
        loss_fn = CrossEntropyLoss()
    elif args.loss_fn == 'DiceLoss':
        loss_fn = diceloss()
    elif args.loss_fn == 'FocalLoss':
        loss_fn = Focal_Loss()
    elif args.loss_fn == 'WDiceLoss':
        loss_fn = WDiceLoss()
    elif args.loss_fn == 'DWDLoss':
        loss_fn = DWDLoss()
    elif args.loss_fn == 'IoULoss':
        loss_fn = IOULoss()
    elif args.loss_fn == 'dice_hd':
        loss_fn = AdaptiveSegLoss(num_classes=4)
    
    # 缩放器
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    Metrics = Evaluate_Metric()
    
    # 日志保存路径
    save_logs_path = f"/root/projects/WS-U2net/U-2-Net/results/logs/{args.model}/L: {args.loss_fn}--S: {args.scheduler}"
    
    if not os.path.exists(save_logs_path):
        os.makedirs(save_logs_path)
    if args.save_flag:
        if args.elnloss:
            log_path = f'{save_logs_path}/optim: {args.optimizer}-lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}/{detailed_time_str}'
            writer = SummaryWriter(log_path)
        else:
            log_path = f'{save_logs_path}/optim: {args.optimizer}-lr: {args.lr}-wd: {args.wd}/{detailed_time_str}'
            writer = SummaryWriter(log_path)
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
            
    # 记录修改后的参数
    if args.elnloss:
        modification_log_name = f"optim: {args.optimizer}-lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}/{detailed_time_str}.md"
    else:
        modification_log_name = f"optim: {args.optimizer}-lr: {args.lr}-wd: {args.wd}/{detailed_time_str}.md"
    params = vars(args)
    params_dict['Parameter'] = printed_params
    params_dict['Value'] = [str(params[p]) for p in printed_params]
    contents = tabulate(params_dict, headers=params_header, tablefmt="grid")

    mdf = os.path.join(save_modification_path, modification_log_name)
    if not os.path.exists(os.path.dirname(mdf)):
        os.makedirs(os.path.dirname(mdf))
    if args.save_flag:
        write_experiment_log.write_exp_logs(mdf, contents) 
    
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

    # 划分数据集
    if args.small_data is not None:
        train_datasets, val_datasets, test_datasets = data_split.small_data_split_to_train_val_test(args.data_path, 
                                                                                                    num_small_data=args.small_data, 
                                                                                                    # train_ratio=0.8, 
                                                                                                    # val_ratio=0.1, 
                            save_path='/root/projects/WS-U2net/U-2-Net/SEM_DATA/CSV',
                            flag=args.split_flag) 
    
    else:
        train_datasets, val_datasets, test_datasets = data_split.data_split_to_train_val_test(args.data_path, train_ratio=train_ratio, val_ratio=val_ratio,
                            save_path='/root/projects/WS-U2net/U-2-Net/SEM_DATA/CSV',   # 保存划分好的数据集路径
                            flag=args.split_flag)

    # 读取数据集
    train_datasets = SEM_DATA(train_datasets, 
                            transforms=SODPresetTrain((256, 256), crop_size=256))
    
    val_datasets = SEM_DATA(val_datasets, 
                            transforms=SODPresetEval((256, 256)))
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataloader = DataLoader(train_datasets, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    val_dataloader = DataLoader(val_datasets, 
                                batch_size=8, 
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=True)
    
    """——————————————————————————————————————————————训练 验证——————————————————————————————————————————————"""
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
  
    best_mean_loss, current_miou = float('inf'), 0.0
    best_epoch = 0 
    patience = 0 
    current_mean_loss = float('inf')
    
    if args.resume:
        start_epoch = checkpoint['epoch']
        print(f"Resume from epoch: {start_epoch}")
       
    for epoch in range(start_epoch, end_epoch):
        
        print(f"✈✈✈✈✈ epoch : {epoch + 1} / {end_epoch} ✈✈✈✈✈✈")
        print(f"--Training-- 😀")
        # 记录时间
        start_time = time.time()
        # 训练
        train_loss, T_OM_loss, T_OP_loss, T_IOP_loss, T_Metric_list = train_one_epoch(model, 
                                                                                    optimizer, 
                                                                                    epoch, 
                                                                                    train_dataloader, 
                                                                                    device=device, 
                                                                                    loss_fn=loss_fn, 
                                                                                    scaler=scaler,
                                                                                    Metric=Metrics,
                                                                                    elnloss=args.elnloss,     #  Elastic Net正则化
                                                                                    l1_lambda=args.l1_lambda,
                                                                                    l2_lambda=args.l2_lambda) # loss_fn=loss_fn, 
        
        # 求平均
        train_OM_loss = T_OM_loss / len(train_dataloader)
        train_OP_loss = T_OP_loss / len(train_dataloader)
        train_IOP_loss = T_IOP_loss / len(train_dataloader)
        train_mean_loss = train_loss / len(train_dataloader)
        
        train_loss_list = [train_OM_loss, train_OP_loss, train_IOP_loss, train_mean_loss]
        # 评价指标 metrics = [recall, precision, dice, f1_score]
        train_metrics ={}
        train_metrics[epoch] = epoch
        train_metrics["Loss"] = train_loss_list
        train_metrics["Recall"] = T_Metric_list[0]
        train_metrics["Precision"] = T_Metric_list[1]
        train_metrics["Dice"] = T_Metric_list[2]
        train_metrics["F1_scores"] = T_Metric_list[3]
        train_metrics["mIoU"] = T_Metric_list[4]
        train_metrics["Accuracy"] = T_Metric_list[5]
        
        # 结束时间
        end_time = time.time()
        train_cost_time = end_time - start_time

        # 打印

        print(
              f"train_OM_loss: {train_OM_loss:.3f}\n"
              f"train_OP_loss: {train_OP_loss:.3f}\n"
              f"train_IOP_loss: {train_IOP_loss:.3f}\n"
              f"train_mean_loss: {train_mean_loss:.3f}\n"
              f"train_cost_time: {train_cost_time:.2f}s\n")
        
        # 验证
        if epoch % args.eval_interval == 0 or epoch == end_epoch - 1:
                      
            e = best_epoch

            print(f"--Validation-- 😀")
            # 记录验证开始时间
            start_time = time.time()
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            mean_loss,OM_loss,OP_loss,IOP_loss, Metric_list = evaluate(model, device, val_dataloader, loss_fn, Metrics) # val_loss, recall, precision, f1_scores

            # 求平均
            val_OM_loss = OM_loss / len(val_dataloader)
            val_OP_loss = OP_loss / len(val_dataloader)
            val_IOP_loss = IOP_loss / len(val_dataloader)
            val_mean_loss = mean_loss / len(val_dataloader)
            val_loss_list = [val_OM_loss, val_OP_loss, val_IOP_loss, val_mean_loss]
            
            # 更新调度器
            scheduler.step()

            # 评价指标 metrics = [recall, precision, dice, f1_score]
            val_metrics ={}
            val_metrics[epoch] = epoch
            val_metrics["Loss"] = val_loss_list
            val_metrics["Recall"] = Metric_list[0]
            val_metrics["Precision"] = Metric_list[1]
            val_metrics["Dice"] = Metric_list[2]
            val_metrics["F1_scores"] = Metric_list[3]
            val_metrics["mIoU"] = Metric_list[4]
            val_metrics["Accuracy"] = Metric_list[5]
            # 验证====结束时间
            end_time = time.time()
            val_cost_time = end_time - start_time

            # 打印结果
            print(
                  f"val_OM_loss: {val_OM_loss:.3f}\n"
                  f"val_OP_loss: {val_OP_loss:.3f}\n"
                  f"val_IOP_loss: {val_IOP_loss:.3f}\n"
                  f"val_mean_loss: {val_mean_loss:.3f}\n"
                  f"val_cost_time: {val_cost_time:.2f}s\n\n")
            
            # 记录日志
            tb = args.tb
            if tb:
                writing_logs(writer, train_metrics, val_metrics, epoch)               
                """-------------------------TXT--------------------------------------------------------"""        
                writer.add_text('val/Metrics', 
                                f"optim: {args.optimizer}, lr: {args.lr}, wd: {args.wd}, l1_lambda: {args.l1_lambda}, l2_lambda: {args.l2_lambda}"+ '\n'
                                f"model: {args.model}, loss_fn: {args.loss_fn}, scheduler: {args.scheduler}"
                                )
                if epoch == 5:
                    run_tensorboard(log_path)               
            
            # 保存指标
            if best_mean_loss >= val_mean_loss:
                best_mean_loss = val_mean_loss
                best_epoch = epoch + 1
            metrics_table_header = ['Metrics_Name', 'Mean', 'OM', 'OP', 'IOP']
            metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']
            epoch_s = f"✈✈✈✈✈ epoch : {epoch + 1} / {end_epoch} ✈✈✈✈✈✈\n"
            model_s = f"model : {args.model} \n"
            lr_s = f"lr : {args.lr} \n"
            wd_s = f"wd : {args.wd} \n"  #####
            dropout_s = f"dropout : {args.dropout_p} \n"
            l1_lambda = f"l1_lambda : {args.l1_lambda} \n"
            l2_lambda = f"l2_lambda : {args.l2_lambda} \n"
            scheduler_s = f"scheduler : {args.scheduler} \n"
            loss_fn_s = f"loss_fn : {args.loss_fn} \n"
            best_epoch_s = f"best_epoch : {best_epoch} \n"
            time_s = f"time : {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} \n"
            cost_s = f"cost_time :{val_cost_time / 60:.2f}mins \n"
            
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
            train_loss_s = f"train_loss : {train_mean_loss:.3f}  🍎🍎🍎\n"
            loss_s = f"val_loss : {val_mean_loss:.3f}   🍎🍎🍎\n"

            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            write_info = epoch_s + model_s + lr_s + wd_s + dropout_s + l1_lambda + l2_lambda + loss_fn_s + scheduler_s + train_loss_s + loss_s + table_s + '\n' + best_epoch_s + cost_s + time_s + '\n'

            # 打印结果
            print(write_info)

            # 保存结果
            save_scores_path = f'{args.save_scores_path}/{args.model}/L: {args.loss_fn}--S: {args.scheduler}'
            
            if args.elnloss:
                results_file = f"optim: {args.optimizer}-lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}/{detailed_time_str}.txt"
            else:
                results_file = f"optim: {args.optimizer}-lr: {args.lr}-wd: {args.wd}/{detailed_time_str}.txt"
            file_path = os.path.join(save_scores_path, results_file)

            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            if args.save_flag:
                with open(file_path, "a") as f:
                    f.write(write_info)                      
       
        if args.save_flag:
            
            # 保存best模型
            if args.elnloss:
                save_weights_path = f"{args.save_weight_path}/{args.model}/L: {args.loss_fn}--S: {args.scheduler}/optim: {args.optimizer}-lr: {args.lr}-l1: {args.l1_lambda}-l2: {args.l2_lambda}/{detailed_time_str}"  # 保存权重路径
            else:
                save_weights_path = f"{args.save_weight_path}/{args.model}/L: {args.loss_fn}--S: {args.scheduler}/optim: {args.optimizer}-lr: {args.lr}-wd: {args.wd}/{detailed_time_str}"
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)

            save_file = {"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "Metrics": Metrics.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_mean_loss": best_mean_loss,
                    "best_epoch": best_epoch,
                    "epoch": epoch,
                    "args": args}
            # 保存当前最佳模型的权重
            best_model_path = f"{save_weights_path}/model_best_ep:{best_epoch}.pth"
            torch.save(save_file, best_model_path)
            print(f"Best model saved at epoch {best_epoch} with mean loss {best_mean_loss}")
            # 删除之前保存的所有包含"model_best"的文件
            path_list = os.listdir(save_weights_path)
            for i in path_list:
                if "model_best" in i and i != f"model_best_ep:{best_epoch}.pth":
                    os.remove(os.path.join(save_weights_path, i))
                    print(f"remove last best weight:{i}")
                            
            current_miou = val_metrics["mIoU"][-1]

            # only save latest 10 epoch weights
            if os.path.exists(f"{save_weights_path}/model_ep:{epoch-10}.pth"):
                os.remove(f"{save_weights_path}/model_ep:{epoch-10}.pth")
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)
            torch.save(save_file, f"{save_weights_path}/model_ep:{epoch}.pth") 
        
        # 记录验证loss是否出现上升       
        if val_mean_loss <= current_mean_loss:
            current_mean_loss = val_mean_loss 
            patience = 0
                     
        else:
            patience += 1 
    
        # 早停判断
        if patience >= 30:
            
            print('恭喜你触发早停！！')
            
            break
        
        
    writer.close()
    total_time = time.time() - initial_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("====training over. total time: {}".format(total_time_str))
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train model on SEM stone dataset")
    
    # 保存路径
    parser.add_argument('--data_path',          type=str, 
                        default="/root/projects/WS-U2net/U-2-Net/SEM_DATA/CSV/rock_sem_chged_256_a50_c80.csv", 
                        help="path to csv dataset")
    
    parser.add_argument('--save_scores_path',   type=str, 
                        default='/root/projects/WS-U2net/U-2-Net/results/save_scores', 
                        help="root path to save scores on training and valing")
    
    parser.add_argument('--save_weight_path',   type=str,
                        default="/root/projects/WS-U2net/U-2-Net/results/save_weights",
                        help="the path of save weights")
    # 模型配置
    parser.add_argument('--model',              type=str, 
                        default="ResD_unet", 
                        help=" unet, ResD_unet, msaf_unet, a_unet, m_unet\
                               Segnet, deeplabv3_resnet50, deeplabv3_mobilenetv3_large, pspnet, u2net_full, u2net_lite,")
    
    parser.add_argument('--loss_fn',            type=str, 
                        default='DiceLoss', 
                        help="'CrossEntropyLoss', 'FocalLoss', 'DiceLoss', 'WDiceLoss', 'DWDLoss', 'IoULoss', 'dice_hd'")
    
    parser.add_argument('--optimizer',          type=str, 
                        default='AdamW', 
                        help="'AdamW', 'SGD' or 'RMSprop'.")
    
    parser.add_argument('--scheduler',          type=str, 
                        default='CosineAnnealingLR', 
                        help="'CosineAnnealingLR', 'ReduceLROnPlateau'.")
    
    # 正则化
    parser.add_argument('--elnloss',        type=bool,  default=False,  help='use elnloss or not')
    parser.add_argument('--l1_lambda',      type=float, default=0.001,  help="L1 factor")
    parser.add_argument('--l2_lambda',      type=float, default=0.001,  help='L2 factor')
    parser.add_argument('--dropout_p',      type=float, default=0.3,    help='dropout rate')
     
    parser.add_argument('--device',         type=str,   default='cuda:0'     )
    parser.add_argument('--resume',         type=str,   default=None,   help="the path of weight for resuming")
    parser.add_argument('--amp',            type=bool,  default=True,   help='use mixed precision training or not')
    
    # flag参数
    parser.add_argument('--tb',             type=bool,  default=True,   help='use tensorboard or not')   
    parser.add_argument('--save_flag',      type=bool,  default=True,   help='save weights or not')    
    parser.add_argument('--split_flag',     type=bool,  default=False,  help='split data or not')
    parser.add_argument('--change_params',  type=bool,  default=False,  help='change params or not')       
    
    # 训练参数
    parser.add_argument('--train_ratio',    type=float, default=0.7     ) 
    parser.add_argument('--val_ratio',      type=float, default=0.1     )
    parser.add_argument('--batch_size',     type=int,   default=8      ) 
    parser.add_argument('--start_epoch',    type=int,   default=0,      help='start epoch')
    parser.add_argument('--end_epoch',      type=int,   default=200,    help='ending epoch')

    parser.add_argument('--lr',             type=float, default=8e-4,   help='learning rate')
    parser.add_argument('--wd',             type=float, default=1e-6,   help='weight decay')
    
    parser.add_argument('--eval_interval',  type=int,   default=1,      help='interval for evaluation')
    parser.add_argument('--small_data',     type=int,   default=None,   help='number of small data')
    parser.add_argument('--Tmax',           type=int,   default=45,     help='the numbers of half of T for CosineAnnealingLR')
    parser.add_argument('--eta_min',        type=float, default=1e-8,   help='minimum of lr for CosineAnnealingLR')
    parser.add_argument('--last_epoch',     type=int,   default=0,      help='start epoch of lr decay for CosineAnnealingLR')

    args = parser.parse_args()
    main(args)