import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

from models.base_model import BaseModel


def _ssim(a, b):
    return structural_similarity_index_measure(a, b, k1=0.001, k2=0.003, data_range=(0.0, 1.0), kernel_size=5)


class Type1Model(BaseModel):

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sar_input, opt_input = batch
        sar_output = self.sar_model(sar_input)
        opt_output = self.opt_model(opt_input)
        return sar_output, opt_output

    def get_reference_image(self, opt_input: torch.Tensor) -> torch.Tensor:
        return self.limiter(opt_input).mean(dim=1, keepdim=True)

    def __criterion(self, sar_output: torch.Tensor, opt_output: torch.Tensor, opt_input: torch.Tensor):
        ref = self.get_reference_image(opt_input)
        mse_sar_opt = F.mse_loss(sar_output, opt_output)
        ssim_sar_opt = _ssim(sar_output.detach(), opt_output)
        ssim_sar_ref = _ssim(sar_output.mean(dim=1, keepdim=True), ref)

        with torch.no_grad():  # not a part of the loss -> no gradients are needed
            ssim_opt_ref = _ssim(opt_output.mean(dim=1, keepdim=True), ref)

        loss = mse_sar_opt + 16 * (1 - ssim_sar_ref) + 3 * (1 - ssim_sar_opt)

        return loss, {'mse_sar_opt': mse_sar_opt.item(), 'ssim_sar_opt': ssim_sar_opt.item(),
                      'ssim_sar_ref': ssim_sar_ref.item(), 'ssim_opt_ref': ssim_opt_ref.item()}

    def _step(self, stage, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        sar_output, opt_output = self.forward(batch)
        sar_input, opt_input = batch
        loss, metrics = self.__criterion(sar_output, opt_output, opt_input)

        metrics[f'loss'] = loss.item()
        metrics = {f'{stage}/{n}': v for n, v in metrics.items()}

        self.log_dict(metrics, prog_bar=True)
        if self.global_step % 1000 == 0 or (stage != 'train' and batch_idx == 1):
            self.save_images(stage, self.trainer.global_step,
                             {
                                 'sar_output': sar_output[0], 'opt_output': opt_output[0], #already in (0, 1) scale
                                 'sar_input': self.preprocess_sar_image(sar_input),
                                 'opt_input': self.preprocess_opt_image(opt_input)
                             })
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step('train', batch, batch_idx)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], bi):
        return self._step('val', batch, bi)


    #
    # def __init__(self, sar_channels: int, opt_channels: int, out_channels: int, sar_use_log_scale: bool = True,
    #              optimizer: OptimizerFactory | None = None,
    #              lr_scheduler: LRSchedulerFactory | None | Literal['default'] = 'default'):
    #     sar_model = self._get_model(sar_channels, out_channels, sar_use_log_scale)
    #     opt_model = self._get_model(opt_channels, out_channels, False)
    #
    #     super().__init__(opt_model, sar_model)
    #     self.lr_scheduler_factory = lr_scheduler
    #     self.optimizer_factory = optimizer
    #
    #
    #     self.limiter = PercentileLimiter(0.5, 99.5)
    #     self.save_hyperparameters()