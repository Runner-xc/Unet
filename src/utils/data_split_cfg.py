import pandas as pd
import os
from pathlib import Path
from .data_augment import aug_data_processing

def data_split_to_train_val_test_cfg(cfg):
    """
    根据配置(cfg)划分数据集为训练集、验证集和测试集，并进行可选的数据增强。

    :param cfg: yacs配置节点对象
    :return: 训练集、验证集、测试集csv文件的路径元组
    """
    data_path = cfg.DATA.DATA_PATH
    flag = cfg.DATA.SPLIT_FLAG
    train_ratio = cfg.DATA.TRAIN_RATIO
    val_ratio = cfg.DATA.VAL_RATIO
    
    root_path = Path(data_path).parent
    # 从配置中获取数据增强版本号
    v = cfg.DATA.AUG.VERSION 
    
    if flag:
        print("Splitting dataset...")
        # 读取数据集
        df = pd.read_csv(data_path)

        # 获取并打乱数据集
        df = df.sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
        
        # 获取数据集的长度
        data_len = len(df)

        # 计算训练集、验证集、测试集的长度
        train_len = int(data_len * train_ratio)
        val_len = int(data_len * val_ratio)
        
        # 划分数据集
        train_data = df.iloc[:train_len]
        val_data = df.iloc[train_len: train_len + val_len]
        test_data = df.iloc[train_len + val_len:]

        # 数据增强 (如果配置中启用了)
        if cfg.DATA.AUG.ENABLE:
            print("Performing data augmentation...")
            train_path_dict = aug_data_processing(
                cfg.DATA.AUG.ROOT_PATH, 
                size=cfg.DATA.IMG_SIZE, # 使用统一的图像尺寸
                aug_times=cfg.DATA.AUG.AUG_TIMES, 
                data_csv=train_data, 
                datasets_name="train", 
                v=v
            )
            train_data = pd.DataFrame(train_path_dict)

            val_path_dict = aug_data_processing(
                cfg.DATA.AUG.ROOT_PATH,
                size=cfg.DATA.IMG_SIZE,
                aug_times=cfg.DATA.AUG.AUG_TIMES,
                data_csv=val_data,
                datasets_name="val",
                v=v
            )
            val_data = pd.DataFrame(val_path_dict)

            test_path_dict = aug_data_processing(
                cfg.DATA.AUG.ROOT_PATH,
                size=cfg.DATA.IMG_SIZE,
                aug_times=cfg.DATA.AUG.AUG_TIMES,
                data_csv=test_data,
                datasets_name="test",
                v=v
            )
            test_data = pd.DataFrame(test_path_dict)

        # 保存划分后的数据集文件
        train_data.to_csv(f"{root_path}/train_{os.path.basename(data_path).split('.')[0]}_{v}.csv", index=False)
        val_data.to_csv(f"{root_path}/val_{os.path.basename(data_path).split('.')[0]}_{v}.csv", index=False)
        test_data.to_csv(f"{root_path}/test_{os.path.basename(data_path).split('.')[0]}_{v}.csv", index=False)
        print("Dataset splitting and saving complete!")

    # 总是返回标准命名格式的文件路径
    train_data_save_path = f"{root_path}/train_{os.path.basename(data_path).split('.')[0]}_{v}.csv"
    val_data_save_path = f"{root_path}/val_{os.path.basename(data_path).split('.')[0]}_{v}.csv"
    test_data_save_path = f"{root_path}/test_{os.path.basename(data_path).split('.')[0]}_{v}.csv"

    return train_data_save_path, val_data_save_path, test_data_save_path


def small_data_split_to_train_val_test_cfg(cfg):
    """
    根据配置(cfg)划分小数据集为训练集、验证集和测试集。主要用于快速调试。

    :param cfg: yacs配置节点对象
    :return: 训练集、验证集、测试集csv文件的路径元组
    """
    data_path = cfg.DATA.DATA_PATH
    num_small_data = cfg.DATA.NUM_SMALL_DATA
    flag = cfg.DATA.SPLIT_FLAG
    train_ratio = cfg.DATA.TRAIN_RATIO
    val_ratio = cfg.DATA.VAL_RATIO
    
    root_path = Path(data_path).parent
    
    if flag:
        print(f"Splitting small dataset (n={num_small_data})...")
        # 读取数据集
        df_small = pd.read_csv(data_path, nrows=num_small_data)
        
        # 打乱数据
        df_small = df_small.sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)

        # 获取数据集的长度
        data_len = len(df_small)

        # 计算训练集、验证集、测试集的长度
        train_len = int(data_len * train_ratio)
        val_len = int(data_len * val_ratio)
        
        # 划分数据集
        train_data = df_small.iloc[:train_len]
        val_data = df_small.iloc[train_len: train_len + val_len]
        test_data = df_small.iloc[train_len + val_len:]

        # 保存数据集
        train_data.to_csv(f"{root_path}/num_{num_small_data}_train_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
        val_data.to_csv(f"{root_path}/num_{num_small_data}_val_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
        test_data.to_csv(f"{root_path}/num_{num_small_data}_test_{os.path.basename(data_path).split('.')[0]}.csv", index=False)
        print("Small dataset splitting complete!")
        
    train_data_save_path = f"{root_path}/num_{num_small_data}_train_{os.path.basename(data_path).split('.')[0]}.csv"
    val_data_save_path = f"{root_path}/num_{num_small_data}_val_{os.path.basename(data_path).split('.')[0]}.csv"
    test_data_save_path = f"{root_path}/num_{num_small_data}_test_{os.path.basename(data_path).split('.')[0]}.csv"

    return train_data_save_path, val_data_save_path, test_data_save_path

if __name__ == '__main__':
    # 这个部分的测试逻辑需要更新，因为它依赖于一个完整的cfg对象
    # 仅作示例
    print("This script is intended to be imported as a module.")
    print("To test, you would need to create a mock cfg object.")
    
    # from config import get_config # 假设config.py在当前目录或上级目录
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='path/to/your/test.yaml')
    # args = parser.parse_args()
    # test_cfg = get_config(args)
    
    # train_csv, val_csv, test_csv = data_split_to_train_val_test(test_cfg)
    # print(f"Train CSV path: {train_csv}")