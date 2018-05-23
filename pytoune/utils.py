import torch


def torch_to_numpy(obj):
    """
    Convert to numpy arrays all tensors inside a Python object composed of the
    supported types.

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the
        tensors are now numpy arrays. Not supported type are left as reference
        in the new object.

    See:
        `pytoune.torch_apply` for supported types.
    """
    return torch_apply(obj, lambda t: t.cpu().detach().numpy())

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
    if isinstance(obj, list) or isinstance(obj, tuple):
        return type(obj)(torch_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k:torch_apply(el, func) for k,el in obj.items()}
    if not torch.is_tensor(obj):
        return obj
    return func(obj)
