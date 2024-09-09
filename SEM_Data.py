import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
"""
1. 保存数据集路径到csv文件
2. 读取数据集，返回张量
"""

def save_sem_paths_to_csv(root_path, csv_path, csv_name) -> pd.DataFrame:
    path_dict = {'img': [], 'mask': []}
    
    # 优化点：添加异常处理，确保目录存在
    if not os.path.isdir(root_path):
        raise ValueError(f"The root path '{root_path}' does not exist or is not a directory.")
    
    # 遍历root_path下的所有目录和文件
    for root, dirs, files in os.walk(root_path):
        # 只在当前层级查找以'im'或'ma'开头的目录
        dirs[:] = [d for d in dirs if d.startswith(('images', 'masks'))]
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 根据目录名前缀分别处理'img'和'mask'目录
            if dir_name.startswith('im'):
                for img_file in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_file)
                    path_dict['img'].append(img_path)
            elif dir_name.startswith('ma'):
                for mask_file in os.listdir(dir_path):
                    mask_path = os.path.join(dir_path, mask_file)
                    path_dict['mask'].append(mask_path)
    
    # 保存DataFrame到CSV文件
    df = pd.DataFrame(path_dict)
    # 确保csv_path目录存在
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path)
    csv_file_path = f"{csv_path}/{csv_name}"
    df.to_csv(csv_file_path, index=False)
    return df

class SEM_DATA(Dataset):
    def __init__(self, data_path, img_transforms=None, mask_transforms=None):
        self.data_path = data_path
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        df = pd.read_csv(self.data_path)
        return len(df)
    
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
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img_array = np.array(img)
        mask_array = np.array(mask)

        # 转为tensor 不能使用ToTensor(),  
        # 1.变成3通道 
        # 2.ToTensor()会对数据做归一化，将[0,255] -> [0, 1] 并且数据类型为float32

        mask_tensor = torch.tensor(mask_array)

        if self.img_transforms is not None:
            img = self.img_transforms(img_array)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask_tensor)

        return img, mask

    def __getitem__(self, index):
        data = self.load_data(index=index)
        return data
          
if __name__ == '__main__':
    #数据集路径
    root_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA'
    csv_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV'
    csv_name = 'SEM_path.csv'
    save_sem_paths_to_csv(root_path, csv_path, csv_name)
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
