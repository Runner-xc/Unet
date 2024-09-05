"""
数据集划分
1. 训练集
2. 验证集
3. 测试集
"""
import pandas as pd

def data_split_to_train_val_test(data_path, train_ratio=0.8, val_ratio=0.1):
    """
    data_path: csv数据集文件路径
    train_ratio: 训练集比例  默认值：0.8
    val_ratio:   验证集比例  默认值：0.1    
    """

    # 读取数据集
    df = pd.read_csv(data_path)

    # 获取数据集的长度
    data_len = len(df)

    # 计算训练集、验证集、测试集的长度
    train_len = int(data_len * train_ratio)
    val_len = int(data_len * val_ratio)
   
    # 划分数据集
    train_data = df.iloc[:train_len]
    val_data = df.iloc[train_len: train_len + val_len]
    test_data = df.iloc[train_len + val_len:]



    return train_data, val_data, test_data

if __name__ == '__main__':
    data_path = "/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/SEM_path.csv"
    train_data, val_data, test_data = data_split_to_train_val_test(data_path)
    print(len(train_data), train_data.shape, train_data['img'])
    
