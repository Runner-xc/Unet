import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
"""
1. 保存数据集路径到csv文件
2. 读取数据集，返回张量
"""

def save_sem_paths_to_csv(root_path, csv_path, csv_name, size) -> pd.DataFrame:
    path_dict = {'img': [], 'mask': []}
    
    # 遍历root_path下的所有目录和文件
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name.startswith('images'):
                dir_path = os.path.join(root, dir_name)
                print(f"当前目录名称：{dir_path}")
                # 检查子目录名称是否包含指定的 size
                if size in os.listdir(dir_path):
                    for img_file in os.listdir(os.path.join(dir_path, size)):
                        img_path = os.path.join(os.path.join(dir_path, size), img_file)
                        path_dict['img'].append(img_path)
            elif dir_name.startswith('masks'):
                dir_path = os.path.join(root, dir_name)
                print(f"当前目录名称：{dir_path}")
                if size in os.listdir(dir_path):
                    for mask_file in os.listdir(os.path.join(dir_path, size)):
                        mask_path = os.path.join(os.path.join(dir_path, size), mask_file)
                        path_dict['mask'].append(mask_path)
        
    # 确保图片和mask对应                
    assert [path_dict['img'][i][:-4] == path_dict['mask'][i][:-4] for i in range(len(path_dict['img']))], "The image and mask paths do not match."
    
    # 保存DataFrame到CSV文件
    df = pd.DataFrame(path_dict)
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path)
    csv_file_path = f"{csv_path}/{csv_name}"
    df.to_csv(csv_file_path, index=False)
    
    return df

class SEM_DATA(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self):
        df = pd.read_csv(self.data_path)
        return len(df)
    
    def __getitem__(self, index):
        data = self.load_data(index=index)
        return data
    
    # 加载数据集
    def load_data(self, index):
        
        df = pd.read_csv(self.data_path)
        img_list = df['img'].tolist()
        mask_list = df['mask'].tolist()
        img_path = img_list[index]
        mask_path = mask_list[index]
        
        # 确保路径存在
        assert os.path.exists(img_path), f"The image path '{img_path}' does not exist."
        assert os.path.exists(mask_path), f"The mask path '{mask_path}' does not exist."
        
        # 读取数据 
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')   # 将mask转换为灰度图像, 不然会多一个维度
        img_array = np.array(img)
        img_name = os.path.basename(img_path)
        # # 归一化图片
        img_array = img_array / 255.0 
        img_array = img_array.astype(np.float32)
        mask_array = np.array(mask)

        if self.transforms is not None:
            data = self.transforms(img_array, mask_array) 
            
        return data, img_name

    
          
if __name__ == '__main__':
    #数据集路径
    root_path = '/mnt/e/VScode/WS-Hub/WS-UNet/UNet/datasets'
    csv_path  = '/mnt/e/VScode/WS-Hub/WS-UNet/UNet/datasets/CSV'
    csv_name  = 'shale_256.csv'
    size = "256"
    save_sem_paths_to_csv(root_path, csv_path, csv_name, size)
    # # 转换
    # img_compose = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.GaussianBlur((3, 3)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.Normalize(mean=[0.485], std=[0.229])
    #     ])
    
    # mask_compose = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomHorizontalFlip(p=0.5)])
    
    # data_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/SEM_path.csv'
    # sem_dataset = SEM_DATA(data_path, img_transforms=img_compose, mask_transforms=mask_compose)
    # print(sem_dataset[1][0].shape, sem_dataset[1][1].shape)
    # trainer = DataLoader(sem_dataset, batch_size=8, shuffle=True)
    # for batch_data in trainer:
    #     img, mask = batch_data
    #     print(img.shape), print(mask.shape)
    #     break
