import lightning as L
from pathlib import Path
import sys
from data.image_folders_data_module import ImageFoldersDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from models.type1 import Type1Model
import torch

torch.set_float32_matmul_precision('high')
CKPT_PATH = Path(sys.argv[1])
DATA_ROOT = Path(sys.argv[2])




def main():
    try:
        model = Type1Model.load_from_checkpoint(CKPT_PATH, strict=False) # Type2Model is weight-compatible with Type1Model
    except:
        model = torch.load(CKPT_PATH, weights_only=False).cuda()
    data = ImageFoldersDataModule(DATA_ROOT, ['s1', 's2'], 1, 32)

    out_path = CKPT_PATH.parent

    trainer = L.Trainer(
        max_epochs=0,
        log_every_n_steps=100,
        logger=TensorBoardLogger(out_path/'logs'),
    )

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
