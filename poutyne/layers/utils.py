import torch.nn as nn


class Lambda(nn.Module):
    """
    Applies a function to the input tensor.

    Args:
        func (Callable[[~torch.Tensor], ~torch.Tensor]): The function to apply.

    Example:

        .. code-block:: python

            # Alternate version to the ``nn.Flatten`` module.
            my_flatten = Lambda(lambda x: x.flatten(1))

    """

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
