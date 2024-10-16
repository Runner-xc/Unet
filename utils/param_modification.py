import argparse

def modify_param(args, modifications, x):
    param_name = modifications[str(x)]
    
    if param_name in ['lr', 'wd', 'l1_lambda', 'l2_lambda', 'dropout_p', 'eta_min']:
        value = float(input(f"请输入你想修改的 {param_name} 值（float）："))
        
    elif param_name in ['eval_interval', 'batch_size', 'Tmax', 'last_epoch', 'small_data']:
        value = int(input(f"请输入你想修改的 {param_name} 值（int）："))
        
    elif param_name in ['save_weights','elnloss', 'split_flag']:
        value = bool(input(f"请输入你想修改的 {param_name} 值（bool）："))
        
    elif param_name in ['optimizer', 'scheduler', 'model', 'loss_fn', 'resume']:
        value = input(f"请输入你想修改的 {param_name} 名字：")
    setattr(args, param_name, value)
    
def param_modification(args, x):
    """
    x: 要修改的参数序号
    """
    modifications = {
        '1': 'lr',
        '2': 'wd',
        '3': 'l1_lambda',
        '4': 'l2_lambda',
        '5': 'elnloss',
        '6': 'dropout_p',
        '7': 'model',
        '8': 'loss_fn',
        '9': 'optimizer',
        '10': 'scheduler',
        '11': 'Tmax',
        '12': 'eta_min',
        '13': 'last_epoch',
        '14': 'save_weights',
        '15': 'batch_size',
        '16': 'small_data',
        '17': 'eval_interval',
        '18': 'split_flag',
        '19': 'resume'
    }   
    
    # 初始化参数
    c_lr = args.lr
    c_l1_lambda = args.l1_lambda
    c_l2_lambda = args.l2_lambda
    c_dropout_p = args.dropout_p
    c_eval_interval = args.eval_interval
    c_batch_size = args.batch_size
    c_optimizer = args.optimizer
    c_small_data = args.small_data
    c_Tmax = args.Tmax
    c_eta_min = args.eta_min
    c_last_epoch = args.last_epoch
    c_save_weights = args.save_weights
    c_scheduler = args.scheduler
    c_model = args.model
    c_loss_fn = args.loss_fn
    c_split_flag = args.split_flag
    
    # 检查并修改参数
    
    while True:
        if x == '0':
            break
        elif str(x) in modifications:
            modify_param(args, modifications, x)
            x = input("如需继续修改，请输入参数序号（1-16），输入 '0' 退出.")
        else:
            x = input("无效的输入，请重新输入:")


    # 构建参数内容字符串
#     content = f"lr:{c_lr}\nl1_lambda:{c_l1_lambda}\nl2_lambda:{c_l2_lambda}\n\
# dropout_p:{c_dropout_p}\neval_interval:{c_eval_interval}\nbatch_size:{c_batch_size}\n\
# optimizer:{c_optimizer}\nsmall_data:{c_small_data}\nTmax:{c_Tmax}\neta_min:{c_eta_min}\n\
# last_epoch:{c_last_epoch}\nsave_weights:{c_save_weights}\nscheduler:{c_scheduler}\n\
# model:{c_model}\nloss_fn:{c_loss_fn}\nsplit_flag:{c_split_flag}"
    
    return args

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
    param_modification(args)