import torch
from torch.autograd import Variable
import numpy as np

def torch_to_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(torch_to_numpy(el) for el in v)
    if isinstance(v, dict):
        return {k:torch_to_numpy(el) for k,el in v}
    if not torch.is_tensor(v):
        return v
    return v.cpu().numpy()

def tensors_to_variables(v, *args, **kwargs):
    if isinstance(v, Variable):
        return v
    if torch.is_tensor(v):
        return Variable(v, *args, **kwargs)
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(tensors_to_variables(el, *args, **kwargs) for el in v)
    if isinstance(v, dict):
        return {k:tensors_to_variables(el, *args, **kwargs) for k,el in v}
    if not torch.is_tensor(v):
        raise ValueError("The type '%s' is not supported by this function." % type(v).__name__)

def variables_to_tensors(v, *args, **kwargs):
    if isinstance(v, Variable):
        return v.data
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(variables_to_tensors(el, *args, **kwargs) for el in v)
    if isinstance(v, dict):
        return {k:variables_to_tensors(el, *args, **kwargs) for k,el in v}
    if not torch.is_tensor(v):
        raise ValueError("The type '%s' is not supported by this function." % type(v).__name__)
