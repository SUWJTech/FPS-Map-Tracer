import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv

kernels_all = [[] for i in range(5)]
num_cycle1 = [1, 2, 3, 4, 5]  

kernels_all2 = [[] for i in range(7)]
num_cycle2 = [1, 2, 3, 4, 5, 6, 7] 

kernels_all3 = [[] for i in range(1)]

kernels_all4 = [[] for i in range(1)]

def GenerateKernels():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle1: 
        kernels = []
        for j in range(i):  
            k_size = (2 * i) + 1  
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2 
            red_size = (j + 1) * 2 - 1
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size 
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val 
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val 

            kernels.append(kernel)
        kernels_all[i - 1] = kernels
        pass
    return kernels_all

def GenerateKernels2():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle2:  
        kernels = []
        for j in range(1): 
            k_size = (2 * i) + 1 
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32) 
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2  
            red_size = (j + 1) * 2 - 1 
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val  
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val  
            kernels.append(kernel)
        kernels_all2[i - 1] = kernels
        pass
    return kernels_all2

def GenerateKernels3():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 9.0
    kernels_all3[0].append(kernel)
    return kernels_all3

def GenerateKernels4():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 8.0 * -1
    kernel[1, 1] = 0
    kernels_all4[0].append(kernel)
    return kernels_all4

kernels = GenerateKernels()
weights = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False)
            for ks in kernels for k in ks
        ]
kernels2 = GenerateKernels3()
weights2 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False)
            for ks in kernels2 for k in ks
        ]
kernels3 = GenerateKernels4()
weights3 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False)
            for ks in kernels3 for k in ks
        ]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MAC(nn.Module):
    def __init__(self, inplanes, outplanes, one=3, two=3, three=3, scales = 4):
        super(MAC, self).__init__()
        if outplanes % scales != 0: 
            raise ValueError('Planes must be divisible by scales')
        self.weights = weights
        self.weights2 = weights2
        self.weights3 = weights3
        self.scales = scales
        self.relu = nn.ReLU(inplace = True)
        self.spx = outplanes // scales
        self.inconv = Conv(inplanes, outplanes, act=False)

        self.conv1 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)
        self.conv1.conv.weight.data = self.weights[one // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv2 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)
        self.conv2.conv.weight.data = self.weights[two // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 1, groups = self.spx),
        )
        self.conv3[0].weight.data = self.weights2[0].repeat(self.spx, 1, 1, 1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 2, groups = self.spx, dilation=2),
        )
        self.conv4[0].weight.data = self.weights3[0].repeat(self.spx, 1, 1, 1)
        
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = Conv(outplanes, outplanes, k=3, act=nn.ReLU)
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.inconv(x)
        input = x
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = self.relu(y + input)
        return output