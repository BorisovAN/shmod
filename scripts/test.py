import lightning as L
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.image_folders_data_module import ImageFoldersDataModule
from lightning.pytorch.loggers import TensorBoardLogger

from models.identity import MSIdentity, RGBIdentity
from models.type1 import Type1Model
import torch

torch.set_float32_matmul_precision('high')



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('DATA_ROOT', type=Path)
    parser.add_argument('CKPT_PATH',  type=Path)
    parser.add_argument('--log-path', type=Path, required=False)
    return parser.parse_args()




def main():

    args = parse_args()

    def get_log_path():
        if args.log_path is not None:
            return args.log_path
        assert str(args.CKPT_PATH) not in ('IDENTITY', 'IDENTITY_RGB')
        if args.CKPT_PATH.parent.name == 'checkpoints':
            return args.CKPT_PATH.parent.parent
        return args.CKPT_PATH.parent

    if str(args.CKPT_PATH) == 'IDENTITY':
        model = MSIdentity(limit_output=True)
    elif str(args.CKPT_PATH) == 'IDENTITY_RGB':
        model = RGBIdentity()
    else:
        try:
            model = Type1Model.load_from_checkpoint(args.CKPT_PATH, strict=False) # Type2Model is weight-compatible with Type1Model
        except:
            model = torch.load(args.CKPT_PATH, weights_only=False).cuda()
    data = ImageFoldersDataModule(args.DATA_ROOT, ['s1', 's2'], 1, 1)

    log_path = get_log_path()

    trainer = L.Trainer(
        max_epochs=0,
        log_every_n_steps=100,
        logger=TensorBoardLogger(log_path, name='test'),
    )

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
