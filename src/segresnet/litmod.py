"""LightningModule for the inpainting baseline. MONAI BasicUNet + composite + masked L1.

Kept deliberately lean — Lightning handles the loop/AMP/checkpointing, MONAI handles the net.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import BasicUNet
from monai.losses import SSIMLoss

from models import UNetAttn3D


def build_net(arch, roi, in_channels=2):
    """arch: 'basicunet' (baseline) | 'unet_w' (winner-faithful 3-down U-Net, PReLU+InstanceNorm+
    dropout0.2) | 'attn' (unet_w + transformer bottleneck) | 'segresnet' | 'dynunet'.
    in_channels: 2 = [voided, mask]; 3 adds a retrieval-prior channel."""
    if arch == "basicunet":
        return BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=1,
                         features=(32, 32, 64, 128, 256, 32))
    if arch == "unet_w":
        # faithful 2025-winner architecture: 3 downsamples, base32 doubling (32->256), no attention.
        return UNetAttn3D(in_channels=2, out_channels=1, feats=(32, 64, 128, 256),
                          dropout=0.2, attn_layers=0)
    if arch == "attn":
        # 4 downsamples -> bottleneck = roi/16 (roi must be divisible by 16, e.g. 208,208,144->13,13,9)
        bn = tuple(int(r) // 16 for r in roi)
        return UNetAttn3D(in_channels=2, out_channels=1, feats=(32, 64, 128, 256, 512),
                          dropout=0.2, attn_layers=4, attn_heads=8, bottleneck_size=bn)
    if arch == "segresnet":
        from monai.networks.nets import SegResNet   # residual encoder-decoder, GroupNorm (robust at bs2)
        return SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=1,
                         init_filters=32, norm=("GROUP", {"num_groups": 8}),
                         blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))   # ~18.8M, in the 10-31M band
    if arch == "dynunet":                              # nnU-Net topology, residual, ~31M (v2 capacity)
        from monai.networks.nets import DynUNet
        return DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=1,
                       kernel_size=[3, 3, 3, 3], strides=[1, 2, 2, 2],
                       upsample_kernel_size=[2, 2, 2], norm_name="INSTANCE",
                       res_block=True, deep_supervision=False)
    raise ValueError(f"unknown arch {arch}")


class LitInpaint(pl.LightningModule):
    def __init__(self, lr=2e-4, features=(32, 32, 64, 128, 256, 32),
                 w_l1=1.0, w_mse=1.0, w_ssim=0.0, ssim_ds=2,
                 arch="basicunet", roi=(128, 128, 128), data_range=1.0, weight_decay=0.0,
                 lr_schedule=None, in_channels=2):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_net(arch, roi, in_channels)
        # ssim_ds: avg-pool factor before the (expensive) 3D SSIM. The full-res 3D sliding window
        # over 128^3 was ~10x slower/step and starved the run of epochs (exp unet_ssim, 12 vs 119
        # epochs -> undertrained, SSIM 0.717). ds=2 -> 64^3 -> ~8x cheaper, near-full epochs.
        # data_range: 1.0 for [0,1] inputs (our norm), 2.0 for [-1,1] (winner norm).
        self.ssim = SSIMLoss(spatial_dims=3, data_range=data_range) if w_ssim > 0 else None

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
            # full-volume (1-SSIM) on the composited pred vs GT: outside the mask pred==voided==y
            # (perfect), so this is a context/realism signal over the hole + its surrounding window.
            # (masking both to the healthy region would corrupt SSIM's sliding window at the seam.)
            # downsample first to keep it cheap (see ssim_ds note in __init__).
            if h.ssim_ds > 1:
                pd = F.avg_pool3d(pred, h.ssim_ds)
                yd = F.avg_pool3d(y, h.ssim_ds)
            else:
                pd, yd = pred, y
            sl = self.ssim(pd, yd)
            loss = loss + h.w_ssim * sl
        if tag == "val" and self.ssim is not None:
            # MASKED SSIM for checkpoint selection — match the official metric: full SSIM map
            # averaged ONLY inside the healthy mask (full-image SSIM is ~0.998 since pred==y
            # outside the mask, so it's useless for selection).
            from torchmetrics.functional import structural_similarity_index_measure as _ssim
            _, smap = _ssim(pred, y, gaussian_kernel=True, return_full_image=True,
                            data_range=float(h.data_range))
            self.log("val_ssim", smap[m].mean(), prog_bar=True, sync_dist=True)
        self.log(f"{tag}_l1", l1, prog_bar=True, sync_dist=True)
        self.log(f"{tag}_mse", mse, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        wd = getattr(self.hparams, "weight_decay", 0.0)
        opt = (torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=wd)
               if wd > 0 else torch.optim.Adam(self.parameters(), lr=self.hparams.lr))
        if getattr(self.hparams, "lr_schedule", None) == "cosine":  # anneal LR->0 over the run
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
