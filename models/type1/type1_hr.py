from .type1 import *


class Type1HRModel(Type1Model):

    def get_reference_image(self, opt_input: torch.Tensor):
        return self.limiter(opt_input[:, (0, 1, 2, 6), ...]).mean(dim=1, keepdim=True)