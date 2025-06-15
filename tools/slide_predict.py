import torch
import numpy as np
from PIL import Image
import math
from tqdm import tqdm
from torchvision import transforms as T

class SlidingWindowPredictor:
    def __init__(self, model, device, window_size=256, stride=128):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.pad_value = 0  # 灰度填充值
        
    def _pad_image(self, image):
        """
        镜像填充图像确保分割完整性
        """
        h, w = image.shape[:2]
        
        pad_h = (self.window_size - (h % self.stride)) % self.window_size
        pad_w = (self.window_size - (w % self.stride)) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, 
                          ((0, pad_h), (0, pad_w), (0, 0)),
                          'reflect')  # 使用镜像填充
        
        return image, (pad_h, pad_w)
    
    def _split_image(self, image):
        """
        生成滑动窗口位置信息
        """
        h, w = image.shape[:2]
        positions = []
        
        num_h = (h - self.window_size) // self.stride + 1
        num_w = (w - self.window_size) // self.stride + 1
        
        for i in range(num_h + 1):
            for j in range(num_w + 1):
                y_start = i * self.stride
                y_end = y_start + self.window_size
                x_start = j * self.stride
                x_end = x_start + self.window_size
                
                # 处理边缘情况
                if y_end > h:
                    y_end = h
                    y_start = h - self.window_size
                if x_end > w:
                    x_end = w
                    x_start = w - self.window_size
                
                positions.append((y_start, y_end, x_start, x_end))
        
        return positions
    
    def predict(self, image_path, save_path):
        """
        执行滑窗预测
        """
        # 加载图像并转换
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        original_tensor = T.ToTensor()(original_image)
        original_tensor = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(original_tensor)
        original_array = torch.permute(original_tensor,(1, 2, 0)).numpy()
        original_h, original_w = original_array.shape[:2]
        
        # 镜像填充
        padded_array, (pad_h, pad_w) = self._pad_image(original_array)
        padded_h, padded_w = padded_array.shape[:2]
        
        # 初始化结果矩阵和权重矩阵
        output = np.zeros((padded_h, padded_w), dtype=np.float32)
        weight = np.zeros((padded_h, padded_w), dtype=np.float32)
        
        # 生成滑动窗口位置
        positions = self._split_image(padded_array)
        
        # 遍历每个窗口
        for y_start, y_end, x_start, x_end in tqdm(positions, desc="Sliding Window"):
            # 提取窗口图像
            crop_img = padded_array[y_start:y_end, x_start:x_end]
            
            # 数据预处理
            img_tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                pred = self.model(img_tensor)
                pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
            
            # 生成权重矩阵（高斯加权）
            window_weight = np.zeros((self.window_size, self.window_size))
            sigma = self.window_size / 4  # 高斯核参数
            for i in range(self.window_size):
                for j in range(self.window_size):
                    d = ((i - self.window_size//2)**2 + (j - self.window_size//2)**2) ** 0.5 
                    window_weight[i,j] = math.exp(-d**2/(2*sigma**2))
            
            # 累加预测结果和权重
            output[y_start:y_end, x_start:x_end] += pred_mask * window_weight
            weight[y_start:y_end, x_start:x_end] += window_weight
        
        # 去除填充区域
        final_output = output[:original_h, :original_w]
        final_weight = weight[:original_h, :original_w]
        
        # 归一化处理
        final_output = np.divide(final_output, final_weight, where=final_weight!=0)
        final_output = np.round(final_output).astype(np.uint8)
        
        # 保存结果
        result_img = Image.fromarray(final_output)
        result_img.save(save_path)
        print(f"Prediction saved to {save_path}")