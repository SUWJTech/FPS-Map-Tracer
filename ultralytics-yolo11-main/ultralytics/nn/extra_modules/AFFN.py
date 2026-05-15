import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False):
        
        super(AFFN, self).__init__()
        self.patch_size = 5
        self.dim = in_features

        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)
        
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        # 自相关融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制频域融合强度
        self.beta = nn.Parameter(torch.tensor(0.5))   # 控制空间域融合强度

    def forward(self, x):
        x = self.project_in(x)
        original_dtype = x.dtype

        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # FFT
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft.to(dtype=Xf.real.dtype)
        # 自相关功率谱
        power = Xf * torch.conj(Xf)          # |X|^2
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        # 融合（频域 + 空间域）
        Xf_new = Xf + self.alpha.to(dtype=Xf.real.dtype) * power     # 频域增强周期结构
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        x_patch_new = x_patch_new + self.beta.to(dtype=R.dtype) * R  # 空间域增强

        # 重组
        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        ).to(dtype=original_dtype)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
