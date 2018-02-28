import torch
from torch.autograd import Variable


def torch_to_numpy(obj):
    """
    Convert to numpy arrays all tensors and variables inside a Python object
    composed of the supported types.

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the
        tensors and variables are now numpy arrays. Not supported type are left
        as reference in the new object.
    """
    if isinstance(obj, Variable):
        obj = obj.data
    if isinstance(obj, list) or isinstance(obj, tuple):
        return type(obj)(torch_to_numpy(el) for el in obj)
    if isinstance(obj, dict):
        return {k:torch_to_numpy(el) for k,el in obj.items()}
    if not torch.is_tensor(obj):
        return obj
    return obj.cpu().numpy()

def tensors_to_variables(obj, *args, **kwargs):
    """
    Convert to variables all tensors inside a Python object composed of the
    supported types.

    Args:
        obj: The Python object to convert.
        *args: The arguments to pass to the Variable constructor.
        **kwargs: The keyword arguments to pass to the Variable constructor.

    Returns:
        A new Python object with the same structure as `obj` but where the
        tensors are now variables.

    Raises:
        ValueError: If a not supported type is inside `obj`.
    """
    if isinstance(obj, Variable):
        return obj
    if torch.is_tensor(obj):
        return Variable(obj, *args, **kwargs)
    if isinstance(obj, list) or isinstance(obj, tuple):
        return type(obj)(tensors_to_variables(el, *args, **kwargs) for el in obj)
    if isinstance(obj, dict):
        return {k:tensors_to_variables(el, *args, **kwargs) for k,el in obj.items()}

    raise ValueError("The type '%s' is not supported by this function." % type(obj).__name__)

def variables_to_tensors(obj):
    """
    Convert to tensors all variables inside a Python object composed of the
    supported types.

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the
        variables are now tensors.

    Raises:
        ValueError: If a not supported type is inside `obj`.
    """
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, Variable):
        return obj.data
    if isinstance(obj, list) or isinstance(obj, tuple):
        return type(obj)(variables_to_tensors(el) for el in obj)
    if isinstance(obj, dict):
        return {k:variables_to_tensors(el) for k,el in obj.items()}

    raise ValueError("The type '%s' is not supported by this function." % type(obj).__name__)
