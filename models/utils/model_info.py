import torch
from thop import profile  

def calculate_computation(model, input_size=(3, 224, 224), device='cuda'):
    """
    计算模型的FLOPs和MACs
    参数：
    - model: 待计算的PyTorch模型
    - input_size: 输入张量尺寸 (channels, height, width)
    - device: 计算设备 ('cuda'/'cpu')
    """
    # 确保模型在评估模式
    model.eval().to(device)
    
    # 生成虚拟输入
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 计算运算量
    flops, params = profile(model, inputs=(dummy_input, ))
    macs = flops / 2  # MACs通常为FLOPs的一半
    
    # 转换单位
    def format_num(num):
        if num > 1e9:
            return f"{num/1e9:.2f} G"
        if num > 1e6:
            return f"{num/1e6:.2f} M"
        return f"{num/1e3:.2f} K"
    
    # 打印结果
    print("="*40)
    print(f"Input size: {input_size}")
    print(f"FLOPs: {format_num(flops)}FLOPs")
    print(f"MACs: {format_num(macs)}MACs")
    print(f"Params: {format_num(params)}")
    print("="*40)

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    from torchvision.models import resnet18
    model = resnet18()
    
    # 计算标准224x224输入的运算量
    calculate_computation(model, (3, 224, 224))
    
    # 计算小尺寸112x112输入的运算量
    calculate_computation(model, (3, 112, 112))