import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Linear(nn.Linear):
    r""" Linear layer for complex number inputs.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__(in_features, out_features, False, device, dtype)

    def forward(self, x):
        x = torch.view_as_real(x).transpose(-2, -1)
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        x = torch.nn.functional.linear(x, weight).transpose(-2, -1)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.contiguous())
        return x

class CirculantAttention(nn.Module):
    r""" Circulant Attention
    https://arxiv.org/abs/2512.21542
    """
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.qkv = Linear(dim, dim * 3)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.SiLU())
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        orig_dtype = x.dtype

        # Prepare Q, K, V, T
        #    (1) qkv=fc(x), qkv=fft(qkv) is mathematically equivalent to x=fft(x), qkv=fc(x)
        #    (2) The latter requires fewer FFT computations, delivering higher throughput
        t = self.gate(x)
        x = x.permute(0, 2, 3, 1) # b c h w -> b h w c
        if x.is_cuda and x.dtype == torch.float16:
            x = x.float()
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Equation 15 of the paper
        #    (1) We use d=1 in practice, as discussed in Table 5
        #    (2) The 1/N factor is implicitly achieved by norm='ortho' in calculating Q, K
        attn = torch.conj(q) * k
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm='ortho')

        # Equation 16 of the paper
        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)
        attn = torch.fft.rfft2(attn, dim=(1, 2))
        x = torch.conj(attn) * v
        x = torch.fft.irfft2(x, s=(h, w), dim=(1, 2), norm='ortho')

        # Output
        x = x.permute(0, 3, 1, 2).to(orig_dtype) * t
        x = self.proj(x)
        return x