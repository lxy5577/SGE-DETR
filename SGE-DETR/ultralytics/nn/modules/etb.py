"""Entangled Transformer Block (ETB) used in SGE-DETR.

Implements the spatial-frequency entangled attention described in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.dwconv2 = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim*4, dim, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
    def forward(self, x):

        x_f = torch.abs(self.weight(torch.fft.fft2(x.float()).real)*torch.fft.fft2(x.float()))
        x_f_gelu = F.gelu(x_f) * x_f

        x_s   = self.dwconv1(x)
        x_s_gelu = F.gelu(x_s) * x_s

        x_f = torch.fft.fft2(torch.cat((x_f_gelu,x_s_gelu),1))
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_f.real) * x_f))

        x_s = self.dwconv2(torch.cat((x_f_gelu,x_s_gelu),1))
        out = self.project_out(torch.cat((x_f,x_s),1))

        return out


def custom_complex_normalization(input_tensor, dim=-1):
    return torch.complex(F.softmax(input_tensor.real, dim=dim), F.softmax(input_tensor.imag, dim=dim))


class Attention_F(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        hidden = max(dim // 16, 1)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True), nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, dim, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.shape
        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())
        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_f = F.normalize(q_f, dim=-1)
        k_f = F.normalize(k_f, dim=-1)
        attn_f = custom_complex_normalization((q_f @ k_f.transpose(-2, -1)) * self.temperature, dim=-1)
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        fft = torch.fft.fft2(x.float())
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(fft.real) * fft))
        return self.project_out(torch.cat((out_f, out_f_l), 1))


class Attention_S(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1conv_1 = nn.Conv2d(dim, dim, 1)
        self.qkv2conv_1 = nn.Conv2d(dim, dim, 1)
        self.qkv3conv_1 = nn.Conv2d(dim, dim, 1)
        self.qkv1conv_3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=dim // 2, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=dim // 2, bias=bias)
        self.conv_3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=dim // 2, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_s = torch.cat((self.qkv1conv_3(self.qkv1conv_1(x)), self.qkv1conv_5(self.qkv1conv_1(x))), 1)
        v_s = torch.cat((self.qkv3conv_3(self.qkv3conv_1(x)), self.qkv3conv_5(self.qkv3conv_1(x))), 1)
        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_s, k_s = F.normalize(q_s, dim=-1), F.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        out_s = rearrange(attn_s.softmax(dim=-1) @ v_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_s_l = torch.cat((self.conv_3(x), self.conv_5(x)), 1)
        return self.project_out(torch.cat((out_s, out_s_l), 1))


class ETB(nn.Module):
    def __init__(self, dim=128, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_S = Attention_S(dim, num_heads, bias)
        self.attn_F = Attention_F(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn_F(self.norm1(x)) + self.attn_S(self.norm1(x))
        return x + self.ffn(self.norm2(x))

