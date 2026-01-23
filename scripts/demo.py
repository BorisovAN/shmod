import sys
from pathlib import Path
import numpy as np
from data.image_folder import ImageFolder
from models.percentile_limiter import PercentileLimiter
from models.type1 import Type1Model
import torch
import cv2
torch.set_float32_matmul_precision('high')


CKPT_PATH = Path(sys.argv[1])

DATA_ROOT = Path(sys.argv[2])

limiter = PercentileLimiter(0.5, 99.5).cuda()


def convert_to_rgb(img: torch.Tensor, *, limit_range: bool = False, use_log_scale: bool = False):
    if use_log_scale:
        img = torch.log10(img + 0.001)

    if limit_range:
        img = limiter(img)

    img = img[0]
    img = torch.clamp(img, 0, 1) * 255
    img = img.to(torch.uint8)
    img = torch.moveaxis(img, 0, -1)  # CHW -> HWC
    img = img.cpu().detach().numpy()
    return img


def show_image(img: np.ndarray, window_name: str, *, _created_windows=set()):
    if window_name not in _created_windows:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        _created_windows.add(window_name)
    cv2.imshow(window_name, img)


def main():
    try:
        model = Type1Model.load_from_checkpoint(CKPT_PATH, strict=False)
    except Exception as E:
        model = torch.load(CKPT_PATH, weights_only=False).cuda()

    model.eval()

    dataset = ImageFolder(DATA_ROOT, ['s1', 's2'])

    assert dataset is not None

    idx = 0
    with torch.no_grad():
        while True:
            print(idx)
            data = tuple(_[None].cuda() for _ in dataset[idx])

            results = model(data)

            s1, s2 = data
            r1, r2 = results

            s1 = convert_to_rgb(s1[:, (0, 1, 0), ...], limit_range=True, use_log_scale=True)
            s2 = convert_to_rgb(s2[:, :3, ...], limit_range=True)
            r1 = convert_to_rgb(r1)
            r2 = convert_to_rgb(r2)

            show_image(s1, 's1')
            show_image(s2, 's2')
            show_image(r1, 'r1')
            show_image(r2, 'r2')

            while True:
                c = cv2.waitKey(0)
                if c == ord('q'):
                    exit(0)
                if c == ord('d'):
                    idx = min(idx + 1, len(dataset) - 1)
                    break
                if c == ord('a'):
                    idx = max(idx - 1, 0)
                    break


if __name__ == "__main__":
    main()
