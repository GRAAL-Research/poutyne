import numpy as np
import torch


def torch_to_numpy(obj, copy=False):
    """
    Convert to Numpy arrays all tensors inside a Python object composed of the
    supported types.

    Args:
        obj: The Python object to convert.
        copy (bool): Whether to copy the memory. By default, if a tensor is
            already on CPU, the Numpy array will be a view of the tensor.

    Returns:
        A new Python object with the same structure as `obj` but where the
        tensors are now Numpy arrays. Not supported type are left as reference
        in the new object.

    Example:
        .. code-block:: python

            >>> from pytoune import torch_to_numpy
            >>> torch_to_numpy({
            ...     'first': torch.tensor([1, 2, 3]),
            ...     'second':[torch.tensor([4,5,6]), torch.tensor([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': array([1, 2, 3]),
                'second': [array([4, 5, 6]), array([7, 8, 9])],
                'third': 34
            }

    See:
        `pytoune.torch_apply` for supported types.
    """
    if copy:
        func = lambda t: t.cpu().detach().numpy().copy()
    else:
        func = lambda t: t.cpu().detach().numpy()
    return torch_apply(obj, func)

def torch_to(obj, *args, **kargs):
    return torch_apply(obj, lambda t: t.to(*args, **kargs))

def torch_apply(obj, func):
    """
    Apply a function to all tensors inside a Python object composed of the
    supported types.

    Supported types are: list, tuple and dict.

    Args:
        obj: The Python object to convert.
        func: The function to apply.

    Returns:
        A new Python object with the same structure as `obj` but where the
        tensors have been applied the function `func`. Not supported type are
        left as reference in the new object.
    """
    fn = lambda t: func(t) if torch.is_tensor(t) else t
    return _apply(obj, fn)

def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k:_apply(el, func) for k, el in obj.items()}
    return func(obj)

def numpy_to_torch(obj):
    """
    Convert to tensors all Numpy arrays inside a Python object composed of the
    supported types.

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the
        Numpy arrays are now tensors. Not supported type are left as reference
        in the new object.

    Example:
        .. code-block:: python

            >>> from pytoune import numpy_to_torch
            >>> numpy_to_torch({
            ...     'first': np.array([1, 2, 3]),
            ...     'second':[np.array([4,5,6]), np.array([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': tensor([1, 2, 3]),
                'second': [tensor([4, 5, 6]), tensor([7, 8, 9])],
                'third': 34
            }


    """
    fn = lambda a: torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    return _apply(obj, fn)
