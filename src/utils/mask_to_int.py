import pandas as pd
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm

def mask_to_int(data_path, save_path, csv_to_int_path):
    data = pd.read_csv(data_path)
    mask_list = data['mask'].tolist()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    updated_paths = []

    mask_list = tqdm(mask_list, desc='=== mask to int ===')
    for idx, path in enumerate(mask_list):
        mask_name = os.path.basename(path)
        mask_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # 将mask转换为整数，使用np.digitize
        mask_array_copy = np.digitize(mask_array, bins=np.unique(mask_array), right=True)

        # 保存修改后的mask
        data_save_path = os.path.join(save_path, mask_name.replace('.png', '.jpg'))
        cv2.imwrite(data_save_path, mask_array_copy.astype('uint8'))  # 确保保存为uint8类型

        # 更新路径
        updated_paths.append(data_save_path)

    # 更新CSV文件
    data['mask'] = updated_paths
    data.to_csv(csv_to_int_path, index=False)

if __name__ == "__main__":
    
    data_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/SEM_path.csv'
    save_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/mask_int'
    csv_to_int_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/SEM_DATA/CSV/SEM_int_path.csv'
    mask_to_int(data_path, save_path, csv_to_int_path)