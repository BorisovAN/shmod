import torch
torch.manual_seed(0)
import torch.optim
from models.type1 import Type1Model
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import *
from pathlib import Path
from data.image_folders_data_module import ImageFoldersDataModule
import sys

EXP_NAME = sys.argv[1]
OUT_PATH = Path(f"../results/{EXP_NAME}")

DATA_ROOT = Path(sys.argv[2])

BASE_LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 128


def main():
    data = ImageFoldersDataModule(DATA_ROOT, ['s1', 's2'], BATCH_SIZE, 32, num_workers=8)

    def get_optimizer(params):
        return torch.optim.NAdam(params, BASE_LR, weight_decay=1e-2)

    def get_scheduler(optimizer):
        return {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, BASE_LR,
                                                             total_steps=EPOCHS * len(data.train_dataloader()),
                                                             div_factor=1e4),
            'interval': 'step',  # 'epoch'
            'frequency': 1
        }

    model = Type1Model(2, 10, 3, True, optimizer=get_optimizer, lr_scheduler=get_scheduler)

    trainer = Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=100,
        logger=TensorBoardLogger('../results/logs'),
        callbacks=[
            ModelCheckpoint(OUT_PATH, "best_{epoch}", monitor='val/loss'),
            ModelCheckpoint(OUT_PATH, filename='last'),
            EarlyStopping('val/loss', patience=8),
            LearningRateMonitor('step')
        ],
        # reload_dataloaders_every_n_epochs=1,
        # limit_train_batches=1024
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
