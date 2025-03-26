"""
数据集划分
1. 训练集
2. 验证集
3. 测试集
"""
import pandas as pd
import os
from .data_augment import aug_data_processing

def data_split_to_train_val_test(aug_args, data_path, flag, train_ratio=0.8, val_ratio=0.1, save_root_path="./datasets"):
    """
    data_path:   csv数据集文件路径
    train_ratio: 训练集比例  默认值：0.8
    val_ratio:   验证集比例  默认值：0.1
    save_root_path:   保存数据集的目录路径，如果不提供，则不保存文件
    flag:        是否划分数据集
    """
    if flag:
        # 读取数据集
        df = pd.read_csv(data_path)

        # 获取数据集的长度
        data_len = len(df)

        # 计算训练集、验证集、测试集的长度
        train_len = int(data_len * train_ratio)
        val_len = int(data_len * val_ratio)
        test_len = data_len - train_len - val_len  

        # 划分数据集
        train_data = df.iloc[:train_len]
        val_data = df.iloc[train_len: train_len + val_len]
        test_data = df.iloc[train_len + val_len:]

        # 数据增强
        train_path_dict = aug_data_processing(aug_args.root_path, size=aug_args.size, aug_times=aug_args.aug_times, data_csv=train_data, datasets_name="train")
        train_data = pd.DataFrame(train_path_dict)

        val_path_dict = aug_data_processing(aug_args.root_path, size=aug_args.size, aug_times=aug_args.aug_times, data_csv=val_data, datasets_name="val")
        val_data = pd.DataFrame(val_path_dict)

        test_path_dict = aug_data_processing(aug_args.root_path, size=aug_args.size, aug_times=aug_args.aug_times, data_csv=test_data, datasets_name="test")
        test_data = pd.DataFrame(test_path_dict)

        # 保存数据集
        train_data.to_csv(f"{save_root_path}/train_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
        val_data.to_csv(f"{save_root_path}/val_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
        test_data.to_csv(f"{save_root_path}/test_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
        print("数据集划分完成！")

    train_data_save_path = f"{save_root_path}/train_{os.path.basename(data_path).split('.')[0]}.csv"
    val_data_save_path = f"{save_root_path}/val_{os.path.basename(data_path).split('.')[0]}.csv"
    test_data_save_path = f"{save_root_path}/test_{os.path.basename(data_path).split('.')[0]}.csv"

    return train_data_save_path, val_data_save_path, test_data_save_path

def small_data_split_to_train_val_test(data_path, num_small_data: int, flag, train_ratio=0.8, val_ratio=0.1, save_root_path="./datasets"):
    """
    data_path: csv数据集文件路径
    train_ratio: 训练集比例  默认值：0.8
    val_ratio:   验证集比例  默认值：0.1
    save_root_path:   保存数据集的目录路径，如果不提供，则不保存文件
    flag:        是否划分数据集
    """
    if flag:
        # 读取数据集
        df_small = pd.read_csv(data_path, nrows=num_small_data + 1)

        # 获取数据集的长度
        data_len = len(df_small)

        # 计算训练集、验证集、测试集的长度
        train_len = int(data_len * train_ratio)
        val_len = int(data_len * val_ratio)
        test_len = data_len - train_len - val_len  

        # 划分数据集
        train_data = df_small.iloc[:train_len]
        val_data = df_small.iloc[train_len: train_len + val_len]
        test_data = df_small.iloc[train_len + val_len:]

        # 如果提供了保存路径，则保存数据集
        if save_root_path:
            train_data.to_csv(f"{save_root_path}/num_{num_small_data}_train_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
            val_data.to_csv(f"{save_root_path}/num_{num_small_data}_val_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
            test_data.to_csv(f"{save_root_path}/num_{num_small_data}_test_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
            print("数据集划分完成！")
        
    train_data_save_path = f"{save_root_path}/num_{num_small_data}_train_{os.path.basename(data_path).split('.')[0]}.csv"
    val_data_save_path = f"{save_root_path}/num_{num_small_data}_val_{os.path.basename(data_path).split('.')[0]}.csv"
    test_data_save_path = f"{save_root_path}/num_{num_small_data}_test_{os.path.basename(data_path).split('.')[0]}.csv"

    return train_data_save_path, val_data_save_path, test_data_save_path

if __name__ == '__main__':
    data_path = "/root/projects/WS-U2net/U-2-Net/SEM_DATA/CSV/SEM_path.csv"
    train_data, val_data, test_data = data_split_to_train_val_test(data_path)
    print(len(train_data), train_data.shape, train_data['img'])
    
