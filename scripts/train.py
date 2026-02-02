import torch

from models.type2 import Type2Model

torch.manual_seed(0)
import torch.optim
from models.type1 import Type1Model
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import *
from pathlib import Path
from data.image_folders_data_module import ImageFoldersDataModule
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--out-path', type=Path, default=Path('../results/'))
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model-type', choices=['type1', 'type2'], default='type1')
    return parser.parse_args()
#
# EXP_NAME = sys.argv[1]
# OUT_PATH = Path(f"../results/{EXP_NAME}")
#
# DATA_ROOT = Path(sys.argv[2])
#
# BASE_LR = 1e-4
# BATCH_SIZE = 16
# EPOCHS = 128


def main():
    args = parse_args()

    data = ImageFoldersDataModule(args.data_root, ['s1', 's2'], args.batch_size, args.batch_size, num_workers=args.num_workers)

    def get_optimizer(params):
        return torch.optim.NAdam(params, args.base_lr, weight_decay=1e-2)

    def get_scheduler(optimizer):
        return {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, args.base_lr,
                                                             total_steps=args.epochs * len(data.train_dataloader()),
                                                             div_factor=1e4),
            'interval': 'step',  # 'epoch'
            'frequency': 1
        }

    def get_model():
        if args.model_type == 'type1':
            return Type1Model(2, 10, 3, True, optimizer=get_optimizer, lr_scheduler=get_scheduler)
        return Type2Model(2, 10, 3, True, optimizer=get_optimizer, lr_scheduler=get_scheduler)

    model = get_model()


    out_path = args.out_path / args.exp_name
    log_path = args.out_path / 'logs'

    trainer = Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=100,
        logger=TensorBoardLogger(log_path),
        callbacks=[
            ModelCheckpoint(out_path, "best_{epoch}", monitor='val/loss'),
            ModelCheckpoint(out_path, filename='last'),
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
