"""LightningModule for the inpainting baseline. MONAI BasicUNet + composite + masked L1.

Kept deliberately lean — Lightning handles the loop/AMP/checkpointing, MONAI handles the net.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import BasicUNet
from monai.losses import SSIMLoss


class LitInpaint(pl.LightningModule):
    def __init__(self, lr=2e-4, features=(32, 32, 64, 128, 256, 32),
                 w_l1=1.0, w_mse=1.0, w_ssim=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = BasicUNet(spatial_dims=3, in_channels=2, out_channels=1, features=features)
        self.ssim = SSIMLoss(spatial_dims=3, data_range=1.0) if w_ssim > 0 else None

    def forward(self, x):
        # x: (B, 2, X, Y, Z) = [voided, mask]; composite so only the hole is learned.
        voided, mask = x[:, :1], x[:, 1:2]
        return self.net(x) * mask + voided * (1 - mask)

    def _step(self, batch, tag):
        x, y, hm = batch
        pred = self(x)
        m = hm > 0.5
        if not m.any():
            return (pred * 0).sum()
        h = self.hparams
        l1 = F.l1_loss(pred[m], y[m])
        mse = F.mse_loss(pred[m], y[m])
        loss = h.w_l1 * l1 + h.w_mse * mse
        if self.ssim is not None:
            loss = loss + h.w_ssim * self.ssim(pred * hm, y * hm)
        self.log(f"{tag}_l1", l1, prog_bar=True, sync_dist=True)
        self.log(f"{tag}_mse", mse, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
