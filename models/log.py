import torch


class Log(torch.nn.Module):

    def __init__(self, eps = 0.01):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor(eps, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        return torch.log(input+self.eps)


class Log10(torch.nn.Module):
    def __init__(self, eps = 0.001):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor(eps, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        result = torch.log10(input+self.eps)
        return result

