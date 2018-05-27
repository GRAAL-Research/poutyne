import torch.nn as nn

class Flatten(nn.Module):
    """
    Flattens the input tensor such that the dimension 0 is the batch.
    Equivalent to ``x.view(x.size(0), -1)``.
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    """
    Returns the same tensor as the input. This module can be useful when
    removing a module from a third-party module.
    """
    def forward(self, x):
        return x

class Lambda(nn.Module):
    """
    Applies a function to the input tensor.

    Args:
        func (Tensor -> Tensor): The function to apply.

    Example:

        .. code-block:: python

            # Alternate version to the ``Flatten`` module.
            my_flatten = Lambda(lambda x: x.view(x.size(0), -1))

    """
    def __init__(self, func):
        self.func = func

    def forward(self, x):
        return self.func(x)
