"""Track B — conditional 3D wavelet diffusion (fastWDM3D-style), on the shared rails.

Same dataset/composite/score as Track A, so the two are directly comparable and ensemble-able.
- Operate in single-level Haar wavelet space (8x reduction -> 3D diffusion fits in memory).
- Palette/concat conditioning: the UNet sees [noisy target wavelet | voided wavelet | mask wavelet]
  (8+8+8 = 24 ch) and predicts the clean target wavelet (prediction_type="sample" -> x0).
- x0-prediction + few-step DDIM => fast sampling (no 1000-step DDPM at inference).
- Loss = wavelet-space x0 MSE (the diffusion objective) + metric-aligned masked recon in image
  space (L1+MSE on the healthy hole of the composited volume) -> matches what we are scored on.
- ALWAYS composite: pred = idwt(x0)*mask + voided*(1-mask), at train and inference.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.losses import SSIMLoss

from wavelet import HaarDWT3D


class LitDiffusion(pl.LightningModule):
    def __init__(self, lr=1e-4, channels=(64, 128, 256, 256),
                 attention_levels=(False, False, True, True), num_res_blocks=2,
                 num_head_channels=32, num_train_timesteps=1000, schedule="linear_beta",
                 infer_steps=4, n_avg=1, w_wave=1.0, w_recon=1.0, w_ssim=0.0, lr_schedule=None,
                 in_channels=24, use_flash_attention=False, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.dwt = HaarDWT3D()
        # in_channels = 8 noisy + 8 voided + 8 mask (=24); 32 adds an 8-ch dwt(prior) conditioner.
        # use_flash_attention: memory-efficient SDPA (no full attn matrix) — needed for large crops
        # (192^3 OOMs the standard softmax); math-identical, so warm-starts existing ckpts unchanged.
        self.net = DiffusionModelUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=8,
            channels=channels, attention_levels=attention_levels,
            num_res_blocks=num_res_blocks, num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention)
        # (1-SSIM) on the composited volume — the lever the 2024/2025 U-Net winners used (with MAE).
        self.ssim = SSIMLoss(spatial_dims=3, data_range=1.0) if w_ssim > 0 else None
        # clip_sample=False: wavelet coeffs are not in [-1, 1], so the default clipping corrupts them.
        sched_kw = dict(num_train_timesteps=num_train_timesteps, schedule=schedule,
                        clip_sample=False, prediction_type="sample")
        self.train_sched = DDPMScheduler(**sched_kw)
        self.ddim = DDIMScheduler(**sched_kw)

    def _cond(self, voided, mask, prior=None):
        cond = torch.cat([self.dwt.dwt(voided), self.dwt.dwt(mask)], dim=1)   # (B, 16, ...)
        if prior is not None:                                                 # + dwt(prior) -> (B, 24, ...)
            cond = torch.cat([cond, self.dwt.dwt(prior)], dim=1)
        return cond

    def _step(self, batch, tag):
        x, y, hm = batch
        voided, mask = x[:, :1], x[:, 1:2]
        prior = x[:, 2:3] if x.shape[1] > 2 else None          # optional 3rd input = structural prior
        x0 = self.dwt.dwt(y)                                   # (B, 8, ...) target wavelet
        cond = self._cond(voided, mask, prior)
        t = torch.randint(0, self.hparams.num_train_timesteps, (x0.shape[0],), device=x0.device)
        xt = self.train_sched.add_noise(x0, torch.randn_like(x0), t)
        pred_x0 = self.net(torch.cat([xt, cond], dim=1), t)    # predicts the clean wavelet
        loss = self.hparams.w_wave * F.mse_loss(pred_x0, x0)
        if self.hparams.w_recon > 0 or self.ssim is not None:
            comp = self.dwt.idwt(pred_x0) * mask + voided * (1 - mask)   # composite (=y outside the hole)
            if self.hparams.w_recon > 0:
                m = hm > 0.5
                if m.any():                                             # MAE+MSE on the scored hole
                    loss = loss + self.hparams.w_recon * (F.l1_loss(comp[m], y[m]) + F.mse_loss(comp[m], y[m]))
            if self.ssim is not None:                                   # (1-SSIM) over the crop (windowed → needs context)
                loss = loss + self.hparams.w_ssim * self.ssim(comp, y)
        self.log(f"{tag}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        wd = getattr(self.hparams, "weight_decay", 0.0)
        opt = (torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=wd) if wd > 0
               else torch.optim.Adam(self.parameters(), lr=self.hparams.lr))
        if getattr(self.hparams, "lr_schedule", None) == "cosine":   # decay to ~0 over the run
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(self.trainer.estimated_stepping_batches))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt

    def _ddim_chunk(self, cond, spatial, device, dtype, k):
        """Run k independent DDIM trajectories (rows of `cond`) -> (k,1,D,H,W) image-space samples."""
        xt = torch.randn((k, 8, *(d // 2 for d in spatial)), device=device, dtype=dtype)
        for ts in self.ddim.timesteps:
            tt = torch.full((k,), int(ts), device=device, dtype=torch.long)
            pred_x0 = self.net(torch.cat([xt, cond], dim=1), tt)
            xt, _ = self.ddim.step(pred_x0, int(ts), xt)
        return self.dwt.idwt(xt)

    @torch.no_grad()
    def sample(self, voided, mask, prior=None, steps=None, n_avg=None, max_batch=32):
        """Few-step DDIM sampling on (B,1,D,H,W) crops -> composited fill at input scale.

        All B x n_avg independent trajectories are flattened and run in batched chunks (up to
        `max_batch`). On CUDA OOM the chunk size HALVES and retries -> uses the fast big batch when the
        GPU allows and degrades gracefully on a smaller one (never crashes). Averaging is unchanged."""
        steps = steps or self.hparams.infer_steps
        n_avg = n_avg or self.hparams.n_avg
        B = voided.shape[0]
        condN = self._cond(voided, mask, prior).repeat_interleave(n_avg, dim=0)   # (B*n_avg, C, ...) 1/traj
        self.ddim.set_timesteps(steps)
        N = B * n_avg
        outs = torch.empty((N, *voided.shape[1:]), device=voided.device, dtype=voided.dtype)
        i, chunk = 0, max_batch
        while i < N:
            k = min(chunk, N - i)
            try:
                outs[i:i + k] = self._ddim_chunk(condN[i:i + k], voided.shape[2:], voided.device, voided.dtype, k)
                i += k
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:            # OOM -> smaller chunk, retry
                if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
                    raise
                torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
        pred = outs.view(B, n_avg, *voided.shape[1:]).mean(dim=1)               # avg n_avg per input
        return pred * mask + voided * (1 - mask)
