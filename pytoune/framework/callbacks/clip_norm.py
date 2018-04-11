from .callbacks import Callback

from torch.nn.utils import clip_grad_norm

class ClipNorm(Callback):
    """
    Uses PyTorch `torch.nn.utils.clip_grad_norm <http://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm>`
    method to clip gradient.

    See:
        `torch.nn.utils.clip_grad_norm <http://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm>`_

    """
    def __init__(self, parameters, max_norm, norm_type=2):
        self.parameters = list(parameters)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, batch):
        clip_grad_norm(self.parameters, self.max_norm, norm_type=self.norm_type)
