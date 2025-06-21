import gradio as gr
import torch
from models import *
from PIL import Image
import numpy as np
from torchvision import transforms as T
import os
import matplotlib.cm as cm
from pathlib import Path
import io
import cv2

WEIGHTS_FOLDER = "./weights"

MODEL_PARAMS = {
    "in_channels": 3,
    "num_classes": 4,
    "base_channels": 32,
    "p": 0
}

model_map = {
    "u2net_full": u2net_full_config(out_ch=4),
    "u2net_lite": u2net_lite_config(out_ch=4),
    "unet": UNet(**MODEL_PARAMS),
    "att_unet": Attention_UNet(**MODEL_PARAMS),
    "res50_unet": RES50_UNet(in_channels=3, num_classes=4),
    "res18_unet": RES18_UNet(in_channels=3, num_classes=4),
    "aw_unet": AWUNet(**MODEL_PARAMS),
    "unetpulsplus": UnetPlusPlus(in_channels=3, num_classes=4, base_channel=32, deep_supervision=False),
    "a_unet": A_UNet(**MODEL_PARAMS),
    "a_unetv4": A_UNetV4(**MODEL_PARAMS),
    "a_unetv6": A_UNetV6(**MODEL_PARAMS),
    "mamba_aunet": Mamba_AUNet(**MODEL_PARAMS),
    "mamba_aunetv2": Mamba_AUNetV2(**MODEL_PARAMS),
    "mamba_aunetv3": Mamba_AUNetV3(**MODEL_PARAMS),
    "mamba_aunetv4": Mamba_AUNetV4(**MODEL_PARAMS),
    "mamba_aunetv5": Mamba_AUNetV5(**MODEL_PARAMS),
    "rdam_unet": RDAM_UNet(**MODEL_PARAMS),
    "aicunet": AICUNet(**MODEL_PARAMS),
    "vm_unet": VMUNet(input_channels=3, num_classes=4),
    "dc_unet": DC_UNet(in_channels=3, num_classes=4, p=0),
    "Segnet": SegNet(num_classes=4, dropout_p=0),
    "pspnet": PSPNet(classes=4, dropout=0, pretrained=False),
    "deeplabv3_resnet50": deeplabv3_resnet50(aux=False, pretrain_backbone=False, num_classes=4),
}

example_images = {
    "示例1": "./images/Image1 - 016.jpeg",
    "示例2": "./images/Image1 - 027.jpeg",
    "示例3": "./images/Image1 - 031.jpeg",
    "示例4": "./images/Image1 - 040.jpeg",
}

def generate_color_map(num_classes):
    colormap = cm.get_cmap("viridis", num_classes)
    colors = [colormap(i)[:3] for i in range(num_classes)]
    return [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]

def load_model(model_name, weight_file_path=None):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型时强制发送到设备
        model = model_map[model_name].to(device)
        
        if weight_file_path is None:
            weights_dir = Path(WEIGHTS_FOLDER) / model_name
            weight_files = list(weights_dir.glob("*.pth"))
            if not weight_files:
                return None, f"未找到默认权重文件：{weights_dir}/*.pth"
            weight_file_path = weight_files[0]
        
        checkpoint = torch.load(weight_file_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, f"✅ {model_name} 加载成功！设备：{device}"
    except Exception as e:
        return None, f"❌ 加载失败：{str(e)}"

def create_error_image(h=512, w=512, text="Error"):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0,0,255), 2, cv2.LINE_AA)
    return img

def inference(image, model):
    try:
        if model is None:
            raise RuntimeError("尚未加载模型")
            
        original_h, original_w = image.shape[:2]
        
        preprocess = T.Compose([
            T.ToTensor(),
            T.Resize((1792, 2048)),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        pil_image = Image.fromarray(image)
        image_tensor = preprocess(pil_image).unsqueeze(0).to(next(model.parameters()).device)
        
        with torch.no_grad():
            output = model(image_tensor)
            if output.dim() == 4:  # [B, C, H, W]
                output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            else:
                output = output.squeeze().cpu().numpy()
        
        colors = generate_color_map(MODEL_PARAMS["num_classes"])
        result = np.zeros((*output.shape, 3), dtype=np.uint8)
        for i in range(MODEL_PARAMS["num_classes"]):
            result[output == i] = colors[i]
        
        # 恢复到原始尺寸
        result = cv2.resize(result, (original_w, original_h), 
                          interpolation=cv2.INTER_NEAREST)
        return result, ""
    
    except Exception as e:
        h, w = (image.shape[:2] if image is not None else (512, 512))
        error_img = create_error_image(h, w, f"Error: {str(e)}")
        return error_img, f"推理错误：{str(e)}"

def load_example_image(example_name):
    image_path = example_images[example_name]
    return np.array(Image.open(image_path))

with gr.Blocks(title="页岩图像分割系统", theme=gr.themes.Default()) as demo:
    current_model = gr.State()
    
    with gr.Tab("模型推理"):
        gr.Markdown("## 页岩图像分割系统")
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1):
                with gr.Group():
                    # 新增回加载图片按钮部分
                    with gr.Column():
                        example_choice = gr.Dropdown(
                            label="选择示例图片",
                            choices=list(example_images.keys()),
                            value="示例1",
                            scale=4
                        )
                        example_btn = gr.Button("加载示例", scale=1, variant="primary") 
                    
                    model_choice = gr.Dropdown(
                        label="选择模型架构",
                        choices=list(model_map.keys()),
                        value="unet"
                    )
                    weight_file = gr.File(label="上传权重文件（可选）", 
                                        file_types=[".pth"])
                    load_btn = gr.Button("加载模型", variant="primary")
                    load_status = gr.Textbox(label="模型状态", interactive=False)
                
            # 右侧图像显示区域
            with gr.Column(scale=2):
                image_input = gr.Image(label="输入图像", type="numpy")
                image_output = gr.Image(label="分割结果")
                
        # 状态显示部分
        inference_error = gr.Textbox(label="错误信息", visible=True)
        # 保留原有的示例加载按钮绑定
        example_btn.click(
            load_example_image,
            inputs=example_choice,
            outputs=image_input
        )
        # 保持其它交互不变...
        example_choice.change(
            load_example_image,
            inputs=example_choice,
            outputs=image_input
        )
        load_btn.click(
            load_model,
            inputs=[model_choice, weight_file],
            outputs=[current_model, load_status]
        )
        # 绑定推理事件
        gr.Button("开始推理", variant="primary").click(
            inference,
            inputs=[image_input, current_model],
            outputs=[image_output, inference_error]
        )

    with gr.Tab("模型信息"):
        gr.Markdown("## 模型详细信息")
        # 待补充模型参数展示组件

if __name__ == "__main__":
    demo.launch()
