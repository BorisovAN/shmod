from .type1 import *


class Type1RGBModel(Type1Model):

    def get_reference_image(self, opt_input: torch.Tensor):
        return self.limiter(opt_input[:, :3, ...]).mean(dim=1, keepdim=True)