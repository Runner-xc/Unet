"""
数据预处理

1. 随机翻转
2. 增加高斯噪声
3. 随机裁剪

"""

import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

"""
随机翻转
"""
class RandomHorizontalFlip():
    def __init__(self, p: float=0.5):
        self.p = p
        self.transform = transforms.RandomHorizontalFlip(p)
    def __call__(self, img):
        if random.random() < self.p:
            img = Image.open(img)
            return img.self.transform(img) # 左右翻转
        
""" 
增加高斯噪声
"""
class GaussianNoise:
    def __init__(self, mean=0, std_dev=5):
        self.mean = mean
        self.std_dev = std_dev
    def __call__(self, img_path):
        # 读取图像
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print("图像读取失败，请检查路径和文件是否存在。")
            return None
        # 将图像转换为 NumPy 数组
        img_array = np.array(img)

        # 生成高斯噪声
        gauss_noise = np.random.normal(self.mean, self.std_dev, img_array.shape).astype(np.uint8)

        # 将高斯噪声添加到图像数组中
        noisy_img_array = img_array + gauss_noise

        # 确保结果在 [0, 255] 范围内
        noisy_img_array = np.clip(noisy_img_array, 0, 255)

        # 将数组转换回 PIL.Image 对象
        noisy_img = Image.fromarray(noisy_img_array.astype('uint8'))
        return noisy_img

"""
随机裁剪
"""
class RandomCrop():
    def __init__(self, size=(320, 320)):  # 确保size是一个元组
        self.size = size
    
    def __call__(self, img_path, num_crops=9):  # 添加num_crops参数来控制裁剪次数
        img = Image.open(img_path)
        w, h = img.size
        th, tw = self.size

        # 存储裁剪后的图像
        crops = []
        for _ in range(num_crops):
            if w == tw and h == th:
                # 如果图片尺寸已经符合要求，则直接添加原图
                crops.append(img)
            else:
                x1 = random.randint(0, w - tw + 1)
                y1 = random.randint(0, h - th + 1)
                img_cropped = img.crop((x1, y1, x1 + tw, y1 + th))
                crops.append(img_cropped)
        
        return crops
# 