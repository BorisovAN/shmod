import torch
from lightning import LightningModule

from models.base_model import compute_test_metrics
from models.log import Log10
from models.percentile_limiter import PercentileLimiter


class MSIdentity(LightningModule):
    def __init__(self, limit_output=False,  sar_use_log_for_sar=True):
        super().__init__()
        self.limiter = PercentileLimiter(0.5, 99.5) if limit_output else None
        self.log10 = Log10() if sar_use_log_for_sar else None
        self.save_hyperparameters()


    def forward(self, batch):
        sar, opt = batch
        B, _, H, W = sar.shape
        out_sar = torch.zeros((B, 3, H, W), dtype=sar.dtype, device=sar.device)

        out_sar[:, (1, 2), ...] = sar[:, (1, 0), ...]
        out_opt = opt[:, :3, ...]
        if self.log10 is not None:
            out_sar = self.log10(out_sar)
        if self.limiter is not None:
            out_sar = self.limiter(out_sar)
            out_opt = self.limiter(out_opt)

        return out_sar, out_opt


    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], bi) :
        sar_output, opt_output = self.forward(batch)
        metrics = compute_test_metrics(sar_output, opt_output)

        _ = {f'test/{n}': v for n, v in metrics.items()}
        metrics = _

        self.log_dict(metrics, prog_bar=True)

class RGBIdentity(LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        sar, opt = batch
        sar = sar.to(torch.float32)/255.0
        opt = opt.to(torch.float32)/255.0
        return sar, opt

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], bi) :
        sar_output, opt_output = self.forward(batch)
        metrics = compute_test_metrics(sar_output, opt_output)

        _ = {f'test/{n}': v for n, v in metrics.items()}
        metrics = _

        self.log_dict(metrics, prog_bar=True)


