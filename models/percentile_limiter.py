import torch
import torch.nn as nn


class PercentileLimiter(nn.Module):

    def __init__(self, pmin: float, pmax: float) -> None:
        super().__init__()
        self.quantiles = nn.Parameter( torch.tensor([pmin/100, pmax/100]) , requires_grad=False)

    @staticmethod
    def to_unit_scale(x, vmin, vmax):
        x = torch.clip(x, vmin, vmax)
        if vmin == vmax:
            x[...] = 0
            return x
        return (x-vmin)/(vmax-vmin)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_batch_dim = x.ndim == 4
        if not has_batch_dim:
            x= x[None]
        x_flat = x.flatten(1, -1)
        qmin, qmax = torch.quantile(x_flat, self.quantiles, -1)

        out = [self.to_unit_scale(x_, min_, max_) for x_, min_, max_ in zip(x, qmin, qmax) ]
        out = torch.stack(out)

        if not has_batch_dim:
            out = out[0]
        return out



if __name__ == "__main__":
    data = torch.rand([32, 10, 256, 256])

    limiter = PercentileLimiter(0.5, 99.5)

    result = limiter(data)

    print(torch.count_nonzero(result==data)/data.flatten().shape[0])

    pass