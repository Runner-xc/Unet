import cv2
import numpy as np
import torch
from model.SED_unet import SED_UNet

# weight_path
weight_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/results/save_weights/SED_unet/L: DiceLoss--S: CosineAnnealingLR/optim: AdamW-lr: 0.0008-wd: 1e-06/2024-10-20_21:19:20/model_best.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化你的模型
model = SED_UNet(in_channels=3, n_classes=4, p=0.5)
model.eval()

# 加载模型权重
load_weight = torch.load(weight_path, map_location="cpu")
if "model" in load_weight:
    model.load_state_dict(load_weight["model"])
else:
    model.load_state_dict(load_weight)

model.to(device)

# 定义滑窗函数
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[1] - window_size[0] + 1, step_size):  # 注意这里的维度顺序
        for x in range(0, image.shape[2] - window_size[1] + 1, step_size):
            yield (x, y, image[:, y:y + window_size[0], x:x + window_size[1]])  # 注意这里的维度顺序

# 加载图片
image_path = '/mnt/c/VScode/WS-Hub/WS-U2net/U-2-Net/Image1 - 003.jpeg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB
image = image.transpose((2, 0, 1))  # 转换为通道优先
image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # 增加批次维度并转换为张量

# 设置窗口大小和步长
window_size = (512, 512)  # 窗口大小
step_size = 256  # 步长

# 创建一个全零数组来存储推理结果
output_image = np.zeros((model.n_classes, image.shape[-2], image.shape[-1]), dtype=np.float32)

# 遍历滑动窗口
for (x, y, window) in sliding_window(image.squeeze(0).cpu().numpy(), window_size, step_size):
    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为张量
    with torch.no_grad():  # 不计算梯度
        prediction = model(window_tensor)
    prediction = prediction.squeeze(0).cpu().numpy()  # 转换回NumPy数组
    output_image[:, y:y + window_size[0], x:x + window_size[1]] = prediction

# 如果需要，可以保存输出图像
cv2.imwrite('output_image.jpg', (output_image * 255).astype(np.uint8))