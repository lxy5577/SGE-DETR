"""Smoke tests for the three paper-level enhancement designs."""
import torch
from ultralytics.nn.extra_modules import ETB, SPDConv, CSPOmniKernel, SNI, GCConv

x = torch.randn(1, 256, 64, 64)
modules = {
    "CPO/SPDConv": SPDConv(256, 256, 2),
    "CPO/CSPOmniKernel": CSPOmniKernel(256, 1),
    "CPO/SNI": SNI(2),
    "ETB": ETB(256, 8),
    "GCConv": GCConv(256, 256, 3, 1, 1),
}
for name, module in modules.items():
    module.eval()
    with torch.no_grad():
        y = module(x)
    print(f"{name:24s}: {tuple(y.shape)}")
