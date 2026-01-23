import torch
import torch.utils.data as D
import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from data.image_folder import ImageFolder, Path

class ImageFoldersDataModule(L.LightningDataModule):
    
    def __init__(self, root: Path, prefixes: list[str], train_batch_size: int, val_batch_size: int, num_workers: int=8,
                 stored_as_channel_first: bool = False,seed: int=42, shuffle_val = False):
        super().__init__()
        assert root.is_dir()
        assert prefixes
        self.generator = torch.Generator().manual_seed(seed)
        self.save_hyperparameters('train_batch_size', 'val_batch_size', 'num_workers')

        def make_dataset(f: Path):
            if f.is_dir():
                return ImageFolder(f, prefixes, stored_as_channel_first)
            else:
                assert not f.exists()
                return None

        self.train_ds = make_dataset(root/'train')
        self.val_ds = make_dataset(root/'val')

        self.val_sampler = D.RandomSampler(self.val_ds, False, generator=torch.Generator().manual_seed(seed)) \
            if (self.val_ds is not None and shuffle_val is True) else None

        self.test_ds = make_dataset(root/'test')


    def train_dataloader(self) -> D.DataLoader:
        assert self.train_ds is not None
        sampler = D.RandomSampler(self.train_ds, True, generator=self.generator)
        return D.DataLoader(self.train_ds, self.hparams.train_batch_size, sampler=sampler,
                            num_workers=self.hparams.num_workers )

    def val_dataloader(self) -> D.DataLoader:
        assert self.val_ds is not None
        return D.DataLoader(self.val_ds, self.hparams.val_batch_size, sampler=self.val_sampler,
                            num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> D.DataLoader:
        assert self.test_ds is not None
        return D.DataLoader(self.test_ds,self.hparams.val_batch_size, False,num_workers=self.hparams.num_workers)







