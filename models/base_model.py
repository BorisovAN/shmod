from typing import Callable, Iterable, Literal
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from lightning.fabric.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from segmentation_models_pytorch import Unet
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio, mean_squared_error

from models.log import Log10
from models.percentile_limiter import PercentileLimiter

OptimizerFactory = Callable[[Iterable], torch.optim.Optimizer]
LRSchedulerFactory = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler | dict]


def _rmse(a: torch.Tensor, b: torch.Tensor):
    mse = (a-b)**2
    mse = mse.mean(dim=(1, 2, 3))
    rmse = torch.sqrt(mse)
    return rmse.mean()


class BaseModel(L.LightningModule):

    @classmethod
    def _get_model(cls, channels_in: int, out_channels: int, add_log: bool,
                   activation: None|Callable[[], nn.Module] = nn.Sigmoid):
        layers = [
            Unet(in_channels=channels_in, encoder_name='efficientnet-b0', encoder_weights=None,
                 decoder_attention_type='scse', classes=out_channels),
        ]
        if add_log:
            layers = [Log10()] + layers
        if activation is not None:
            layers = layers + [activation()]
        return nn.Sequential(*layers)

    def __init__(self, sar_channels: int, opt_channels: int, out_channels: int, sar_use_log_scale: bool = True,
                 optimizer: OptimizerFactory | None = None,
                 lr_scheduler: LRSchedulerFactory | None | Literal['default'] = 'default'):
        super().__init__()
        self.sar_model = self._get_model(sar_channels, out_channels, sar_use_log_scale)
        self.opt_model = self._get_model(opt_channels, out_channels, False)


        self.lr_scheduler_factory = lr_scheduler
        self.optimizer_factory = optimizer

        self.limiter = PercentileLimiter(0.5, 99.5)
        self.save_hyperparameters()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.NAdam(self.parameters(), 2e-4, weight_decay=1e-2)

        if self.lr_scheduler_factory is None:
            return optimizer

        if isinstance(self.lr_scheduler_factory, str) and self.lr_scheduler_factory == 'default':
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-4, total_iters=4)
        else:
            lr_scheduler = self.lr_scheduler_factory(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def compute_test_metrics(self, sar_output: torch.Tensor, opt_output: torch.Tensor):
        return {
            'ssim': structural_similarity_index_measure(sar_output, opt_output, data_range=(0.0, 1.0), kernel_size=5),
            'rmse': _rmse(sar_output, opt_output),
            'psnr': peak_signal_noise_ratio(opt_output, sar_output, (0.0, 1.0), dim=(1, 2, 3))
        }

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sar_input, opt_input = batch
        sar_output = self.sar_model(sar_input)
        opt_output = self.opt_model(opt_input)
        return sar_output, opt_output

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], bi) :
        sar_output, opt_output = self.forward(batch)
        metrics = self.compute_test_metrics(sar_output, opt_output)

        _ = {f'test/{n}': v for n, v in metrics.items()}
        metrics = _

        self.log_dict(metrics, prog_bar=True)

    @torch.inference_mode()
    def preprocess_sar_image(self, image: torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        if self.hparams['sar_use_log_scale']:
            image = torch.log10(image+0.001)
        if image.ndim == 3 and image.shape[0] > 1:
            image = image[(0, 1, 0), ...]
        image = self.limiter(image)
        return image

    @torch.inference_mode()
    def preprocess_opt_image(self, image: torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        image = image[:3]
        image = self.limiter(image)
        return image
    @torch.inference_mode()
    def save_images(self, prefix: str, step: int | None, images: dict[str, torch.Tensor]):
        if not isinstance(self.logger, TensorBoardLogger):
            return

        # noinspection PyUnresolvedReferences
        tb_logger: SummaryWriter = self.logger.experiment


        for name in images:
            image = images[name]
            assert image.ndim in (2, 3)

            tb_logger.add_image(f'{prefix}/{name}', images[name], step)


if __name__ == "__main__":
    torch.manual_seed(0)
    t1 = torch.rand([32, 3, 16, 16]).cuda()
    t2 = torch.rand([32, 3, 16, 16]).cuda()

    psnr1 = peak_signal_noise_ratio(t1, t2, data_range=(0.0, 1.0))
    psnr2 = peak_signal_noise_ratio(t1, t2,data_range=(0.0, 1.0), dim=(1, 2, 3))

    mse1 = torch.sqrt(F.mse_loss(t1, t2))
    mse2 = torch.sqrt(F.mse_loss(t1, t2, reduction='none').mean(dim=(1, 2, 3)).mean())
    pass

