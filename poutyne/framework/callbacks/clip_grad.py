from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from .callbacks import Callback

class ClipNorm(Callback):
    """
    Uses PyTorch `torch.nn.utils.clip_grad_norm_
    <http://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_
    method to clip gradient.

    See:
        `torch.nn.utils.clip_grad_norm_
        <http://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_

    """
    def __init__(self, parameters, max_norm, *, norm_type=2):
        super().__init__()
        self.parameters = list(parameters)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, batch):
        clip_grad_norm_(self.parameters, self.max_norm, norm_type=self.norm_type)

class ClipValue(Callback):
    """
    Uses PyTorch `torch.nn.utils.clip_grad_value_
    <http://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_value_>`_
    method to clip gradient.

    See:
        `torch.nn.utils.clip_grad_value_
        <http://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_value_>`_

    """
    def __init__(self, parameters, clip_value):
        super().__init__()
        self.parameters = list(parameters)
        self.clip_value = clip_value

    def on_backward_end(self, batch):
        clip_grad_value_(self.parameters, self.clip_value)
