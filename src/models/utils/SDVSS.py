import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 模块1: 空间参数生成器
# ==============================================================================
class SpatialParameterGenerator(nn.Module):
    def __init__(self, dim, hidden_ratio=0.5):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2 * dim, kernel_size=1)
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 模块2: 空间动态VSSM模块
# ==============================================================================
class SpatiallyDynamicVSSMBlock(nn.Module):
    def __init__(self, dim, ssm_d_state=16, ssm_d_conv=3):
        super().__init__()
        self.dim = dim
        self.d_state = ssm_d_state
        self.d_conv = ssm_d_conv

        self.param_generator = SpatialParameterGenerator(dim)

        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim,
            kernel_size=ssm_d_conv,
            padding=(ssm_d_conv - 1) // 2,
            groups=dim,
        )
        self.x_proj = nn.Linear(dim, 1 + 2 * self.d_state, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.A_log = nn.Parameter(torch.randn(dim, self.d_state))

    def selective_scan(self, x, delta, A, B, C):
        B_batch, L_len, C_dim = x.shape
        N_state = self.d_state
        delta_A = torch.exp(delta.unsqueeze(-1) * self.A_log.unsqueeze(0).unsqueeze(0))
        B = B.view(B_batch, L_len, 1, N_state)
        h = torch.zeros(B_batch, C_dim, N_state, device=x.device)
        ys = []
        for i in range(L_len):
            h = delta_A[:, i] * h + B[:, i] * x[:, i].unsqueeze(-1)
            y = (h * C[:, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        return torch.stack(ys, dim=1)

    def ssm_pass(self, scanned_x):
        x_conv = self.conv1d(scanned_x.transpose(1, 2)).transpose(1, 2)
        x_activated = F.silu(x_conv)
        x_for_ssm = self.x_proj(x_activated)
        delta, B, C = x_for_ssm.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)
        y = self.selective_scan(x_activated, delta, self.A_log, B, C)
        return self.out_proj(y)

    def forward(self, x):
        B, C, H, W = x.shape
        modulation_maps = self.param_generator(x)
        scale_map, bias_map = modulation_maps.chunk(2, dim=1)

        x_forward = x.flatten(2).permute(0, 2, 1)
        scale_forward = scale_map.flatten(2).permute(0, 2, 1)
        bias_forward = bias_map.flatten(2).permute(0, 2, 1)

        x_backward = x.flip(dims=[2, 3]).flatten(2).permute(0, 2, 1)
        scale_backward = scale_map.flip(dims=[2, 3]).flatten(2).permute(0, 2, 1)
        bias_backward = bias_map.flip(dims=[2, 3]).flatten(2).permute(0, 2, 1)

        y_forward = self.ssm_pass(x_forward)
        y_backward = self.ssm_pass(x_backward)
        
        y_forward_modulated = y_forward * (1 + scale_forward) + bias_forward
        y_backward_modulated = y_backward * (1 + scale_backward) + bias_backward

        y_backward_final = y_backward_modulated.transpose(1, 2).view(B, C, H, W).flip(dims=[2, 3])
        y_forward_final = y_forward_modulated.transpose(1, 2).view(B, C, H, W)
        
        output = y_forward_final + y_backward_final
        return output + x

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn

class SDVSS(nn.Module):
    """
    Structured D-State Vision Space (SDVSS) block.
    This version uses the high-level mamba_inner_fn to avoid low-level call issues.
    """
    def __init__(
        self, 
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 将所有参数生成步骤集成到一个大的线性层，这是 mamba_inner_fn 的推荐做法
        # 维度: d_inner (for z) + d_inner (for x) + dt_rank (for delta) + d_state (for B) + d_state (for C)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2 + self.dt_rank + 2 * self.d_state, bias=bias, **factory_kwargs)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # dt 的投影
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # 初始化 dt 投影的权重和偏置
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.data.copy_(inv_dt)

        # State space 矩阵 A 和 D
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
    def forward(self, x):
        # x 的形状: (B, L, D)
        _, L, _ = x.shape

        # 1. 一次性投影得到所有参数
        xz_dt_BC = self.in_proj(x)
        
        # 2. 分离出 z, x, delta, B, C
        z, x, delta, B, C = torch.split(
            xz_dt_BC, 
            [self.d_inner, self.d_inner, self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # 3. 使用 mamba_inner_fn
        # 这个函数在内部处理了 1D 卷积, SiLU 激活, delta 计算, 和 selective scan
        y = mamba_inner_fn(
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            z,
            delta,
            self.dt_proj.weight,
            self.dt_proj.bias,
            self.A_log,
            B,
            C,
            self.D,
            delta_softplus=True
        )
        
        # 4. 输出投影
        output = self.out_proj(y)
        
        return output
# ==============================================================================
# 测试代码 
# ==============================================================================
if __name__ == '__main__':
     # 配置参数
    batch_size = 4
    seq_len = 64  # 序列长度, 比如 H*W
    d_model_dim = 96 # 模型维度 C
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running test on device: {device}")

    # 创建一个测试输入张量
    test_input = torch.randn(batch_size, seq_len * seq_len, d_model_dim, device=device)
    block = SDVSS(
        d_model=d_model_dim, 
        d_state=16, 
        d_conv=4, 
        expand=2,
        device=device
    )
    block.to(device)

    # 前向传播
    print(f"Input shape: {test_input.shape}")
    output = block(test_input)

    # 检查输出形状
    print(f"Output shape: {output.shape}")

