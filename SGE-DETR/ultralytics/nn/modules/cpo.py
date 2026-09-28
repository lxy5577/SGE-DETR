"""Collaborative P2-P3 Optimization (CPO).

CPO corresponds to the SOEP-RFPN design in the original experiments and
combines SPDConv, CSPOmniKernel, and SNI for small-object feature enhancement.
"""
import torch
import torch.nn as nn

from ..modules.conv import Conv

class SPDConv(nn.Module):
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)


class FGM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out = torch.abs(torch.fft.ifft2(x1 * x2_fft, dim=(-2, -1), norm='backward'))
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ker, pad = 31, 15
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.dw_13 = nn.Conv2d(dim, dim, (1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, (ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, 1)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fca = torch.abs(torch.fft.ifft2(x_att * x_fft, dim=(-2, -1), norm='backward'))
        x_sca = self.conv(self.pool(x_fca)) * x_fca
        x_sca = self.fgm(x_sca)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        return self.out_conv(self.act(out))


class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * e))

    def forward(self, x):
        c = int(x.size(1) * self.e)
        ok_branch, identity = torch.split(self.cv1(x), [c, x.size(1) - c], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


class SNI(nn.Module):
    def __init__(self, up_f=2):
        super().__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1 / (up_f ** 2)

    def forward(self, x):
        return self.alpha * self.us(x)



class GSConvE(nn.Module):
    """GSConv enhancement used only by the standalone SOEP-RFPN reference config."""
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.GELU(),
        )
    def forward(self, x):
        x1 = self.cv1(x); x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
