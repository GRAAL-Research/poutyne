from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from .callbacks import Callback


class ClipNorm(Callback):
    """
    Uses PyTorch's :func:`~torch.nn.utils.clip_grad_norm_()`
    method to clip gradient.

    See:
        :func:`torch.nn.utils.clip_grad_norm_()`

    """
    def __init__(self, parameters, max_norm, *, norm_type=2):
        super().__init__()
        self.parameters = list(parameters)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, batch_number):
        clip_grad_norm_(self.parameters, self.max_norm, norm_type=self.norm_type)


class ClipValue(Callback):
    """
    Uses PyTorch's :func:`~torch.nn.utils.clip_grad_value_()`
    method to clip gradient.

    See:
        :func:`torch.nn.utils.clip_grad_value_()`

    """
    def __init__(self, parameters, clip_value):
        super().__init__()
        self.parameters = list(parameters)
        self.clip_value = clip_value

    def on_backward_end(self, batch_number):
        clip_grad_value_(self.parameters, self.clip_value)
