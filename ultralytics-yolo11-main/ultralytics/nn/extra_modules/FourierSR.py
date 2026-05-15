import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierSR(nn.Module):
    """
    channels: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """
    def __init__(self, in_channels, out_channels, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        assert in_channels % num_blocks == 0, f"in_channels {in_channels} should be divisble by num_blocks {num_blocks}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = in_channels // self.num_blocks
        self.scale = 0.02

        self.w = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        weight = torch.view_as_complex(self.w.float().contiguous())
        x = torch.einsum('bkihw,kio->bkohw', x, weight)

        w1, w2, b = self.w1.float(), self.w2.float(), self.b.float()
        o1_real = F.relu(
            torch.mul(x.real, w1[0].unsqueeze(dim=0)) - \
            torch.mul(x.imag, w1[1].unsqueeze(dim=0)) + \
            b[0, :, :, None, None]
        ) # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]
        
        o1_imag = F.relu(
            torch.mul(x.imag, w2[0].unsqueeze(dim=0)) + \
            torch.mul(x.real, w2[1].unsqueeze(dim=0)) + \
            b[1, :, :, None, None]
        ) # [16, 8, 8, 48, 25] x.real=[16, 8, 8, 48, 25]

        x = torch.stack([o1_real, o1_imag], dim=-1) # [16, 8, 8, 48, 25, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) # [16, 8, 8, 48, 25]
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return self.conv_1x1(x + bias)