"""Single-level orthonormal 3D Haar wavelet transform (the WDM front-end).

fastWDM3D-style: diffusion runs in wavelet space, where one Haar level gives an 8x volume
reduction (240^3 -> 8 x 120^3), so 3D diffusion fits in GPU memory. Critically sampled +
orthonormal => the inverse is just a transposed conv with the same kernels (perfect recon).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_kernels():
    """8 separable 2x2x2 Haar kernels = all {low, high} combinations over the 3 axes."""
    L = torch.tensor([1.0, 1.0]) / 2 ** 0.5
    H = torch.tensor([1.0, -1.0]) / 2 ** 0.5
    f = (L, H)
    ks = [torch.einsum("i,j,k->ijk", f[a], f[b], f[c])
          for a in range(2) for b in range(2) for c in range(2)]
    return torch.stack(ks, 0).unsqueeze(1)            # (8, 1, 2, 2, 2)


class HaarDWT3D(nn.Module):
    """dwt: (B,1,D,H,W) -> (B,8,D/2,H/2,W/2);  idwt: the exact inverse. D,H,W must be even."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", _haar_kernels())

    def dwt(self, x):
        return F.conv3d(x, self.w, stride=2)

    def idwt(self, c):
        return F.conv_transpose3d(c, self.w, stride=2)
