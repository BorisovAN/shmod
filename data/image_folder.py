from typing import Sequence
from pathlib import Path
import torch
import tifffile
import cv2

class ImageFolder(torch.utils.data.Dataset):

    """
    Dataset that represents a folder with grouped images from different modalities.

    Images must be named according to template
      <prefix><id>.<extension>.
    where prefix defines the modality;
          id is used to match images of different modalities (i.e. defines a territory);
          extension is a common image extension.

    Id is essentially a part of a filename that is left after trimming both prefix and extension.
    There is no limitations on prefix or id content.


    """
    TIFF_EXTENSIONS = ['.tif', '.tiff']
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png',  '.bmp', '.gif']+TIFF_EXTENSIONS


    @classmethod
    def is_image_file(cls, filename: Path):
        return filename.is_file() and filename.suffix.lower() in cls.IMAGE_EXTENSIONS

    @classmethod
    def read_image(cls, filename: Path, stored_as_channels_first: bool = False) -> torch.Tensor:
        assert cls.is_image_file(filename)
        try:
            if filename.suffix.lower() in cls.TIFF_EXTENSIONS:

                result = tifffile.imread(filename)
            else:
                result = cv2.imread(str(filename))

            result = torch.from_numpy(result).float()

            if len(result.shape) > 2:

                if not stored_as_channels_first:
                    result = torch.moveaxis(result, -1, 0).contiguous()

            return result
        except Exception as e:
            e.add_note(f'Error when reading {filename}')
            raise e


    def __init__(self, root: Path, prefixes: Sequence[str], stored_as_channels_first: bool = False):
        """
        creates an ImageFolder instance

        The folder must contain images named <prefix><id>.<extension>.
        Prefix defines the modality.
        Id is used to match images of different modalities.

        :param root: path to the folder.
        :param prefixes: List of prefixes to load
        :param stored_as_channels_first:  indicate that images are already stored in CHW-format and do not need a rearrangement of dimensions
        """

        self.stored_as_channels_first = stored_as_channels_first
        assert root.is_dir()
        assert len(prefixes) > 0

        def has_prefix(name: str):
            for prefix in prefixes:
                if name.startswith(prefix):
                    return True
            return False

        files: list[Path] = [f for f in root.glob('*') if self.is_image_file(f) and has_prefix(f.name)]

        def trim_prefix_and_extension(prefix: str, path: Path):
            stem = path.stem
            result = stem.removeprefix(prefix)
            return result


        files_by_prefix = {}
        files_and_prefixes_by_keys = {}

        for p in prefixes:
            files_by_prefix[p] = [f for f in files if f.name.startswith(p)]

        for p in files_by_prefix:
            files_by_keys = {  trim_prefix_and_extension(p, f):f for f in files_by_prefix[p] }
            for k in files_by_keys:
                if k not in files_and_prefixes_by_keys:
                    files_and_prefixes_by_keys[k] = [files_by_keys[k]]
                else:
                    files_and_prefixes_by_keys[k].append(files_by_keys[k])

        result = []

        for k in files_and_prefixes_by_keys:
            if len(files_and_prefixes_by_keys[k]) != len(prefixes):
                continue
            result.append(files_and_prefixes_by_keys[k])

        self.image_sets = sorted(result, key=lambda i: i[0].name)

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, idx):

        image_set = self.image_sets[idx]

        result = []
        for f in image_set:
            i = self.read_image(f, self.stored_as_channels_first)
            result.append(i)
        return tuple(result)


if __name__ == "__main__":

    ds =ImageFolder(Path('/mnt/hot_data/datasets/multisen_ge/test'), ["s1", "s2"])

    #exit(0)

    out_path = Path('/mnt/hot_data/datasets/multisen_ge/test_remapped')
    out_path.mkdir()
    for i in range(len(ds)):
        s1_path, s2_path = ds.image_sets[i]

        s1_new_path = out_path / f's1_{i}.tif'
        s2_new_path = out_path / f's2_{i}.tif'

        s1_new_path.hardlink_to(s1_path)
        s2_new_path.hardlink_to(s2_path)














