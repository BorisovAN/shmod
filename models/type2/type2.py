import torch
import torch.nn.functional as F
from models.base_model import BaseModel, OptimizerFactory, LRSchedulerFactory, Literal
from models.log import Log10
from torchmetrics.functional import structural_similarity_index_measure


def _ssim_limited(a, b):
    return structural_similarity_index_measure(a, b, k1=0.001, k2=0.003, data_range=(0.0, 1.0), kernel_size=5)


def _ssim_full_range(a, b):
    return structural_similarity_index_measure(a, b)


class Type2Model(BaseModel):

    def __init__(self, sar_channels: int, opt_channels: int, out_channels: int, sar_use_log_scale: bool = True,
                 optimizer: OptimizerFactory | None = None,
                 lr_scheduler: LRSchedulerFactory | None | Literal['default'] = 'default'):
        super().__init__(sar_channels, opt_channels, out_channels, sar_use_log_scale, optimizer, lr_scheduler)

        self.sar_inv_model = self._get_model(out_channels, sar_channels, False, activation=None)
        self.opt_inv_model = self._get_model(out_channels, opt_channels, False, activation=None)
        if sar_use_log_scale:
            self.log10 = Log10()
        else:
            self.log10 = None

    def forward_inv(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sar, opt = batch
        sar_out = self.sar_inv_model(sar)
        opt_out = self.opt_inv_model(opt)
        return sar_out, opt_out

    def __criterion(self, sar_output: torch.Tensor, opt_output: torch.Tensor,
                    sar_rec: torch.Tensor, opt_rec: torch.Tensor,
                    sar_input: torch.Tensor, opt_input: torch.Tensor):
        if self.hparams['sar_use_log_scale']:
            sar_input = self.log10(sar_input)

        feature_mse = F.mse_loss(sar_output, opt_output)
        feature_ssim = _ssim_limited(sar_output, opt_output)

        sar_rec_mse = F.mse_loss(sar_input, sar_rec)
        sar_rec_ssim = _ssim_full_range(sar_input, sar_rec)

        opt_rec_mse = F.mse_loss(opt_input, opt_rec)
        opt_rec_ssim = _ssim_full_range(opt_input, opt_rec)

        loss = 3 * feature_mse + 8 * (1 - feature_ssim) \
               + sar_rec_mse + (1 - sar_rec_ssim) \
               + opt_rec_mse + (1 - opt_rec_ssim)

        return loss, {
            'feature_mse': feature_mse.item(),
            'feature_ssim': feature_ssim.item(),
            'sar_rec_mse': sar_rec_mse.item(),
            'sar_rec_ssim': sar_rec_ssim.item(),
            'opt_rec_mse': opt_rec_mse.item(),
            'opt_rec_ssim': opt_rec_ssim.item()
        }

    def _step(self, stage, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        out = self.forward(batch)
        rec = self.forward_inv(out)

        loss, metrics = self.__criterion(*out, *rec, *batch)

        metrics[f'loss'] = loss.item()
        metrics = {f'{stage}/{n}': v for n, v in metrics.items()}

        sar_output, opt_output = out
        sar_input, opt_input = batch
        sar_rec, opt_rec = rec

        self.log_dict(metrics, prog_bar=True)
        if self.global_step % 1000 == 0 or (stage != 'train' and batch_idx == 1):
            self.save_images(stage, self.trainer.global_step,
                             {
                                 'sar_output': self.preprocess_opt_image(sar_output),
                                 'opt_output': self.preprocess_opt_image(opt_output),

                                 'sar_input': self.preprocess_sar_image(sar_input),
                                 'opt_input': self.preprocess_opt_image(opt_input),

                                 'sar_rec': self.preprocess_sar_image(sar_rec),
                                 'opt_rec': self.preprocess_opt_image(opt_rec)
                             }
                             )
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step('train', batch, batch_idx)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], bi):
        return self._step('val', batch, bi)
