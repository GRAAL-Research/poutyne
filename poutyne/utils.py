"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

# -*- coding: utf-8 -*-
import os
import random
import numbers
from typing import IO, Any, BinaryIO, Union
import warnings

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import Dataset

from .warning_manager import warning_settings


def torch_to_numpy(obj, copy=False):
    """
    Convert to Numpy arrays all tensors inside a Python object composed of the supported types.

    Args:
        obj: The Python object to convert.
        copy (bool): Whether to copy the memory. By default, if a tensor is already on CPU, the
            Numpy array will be a view of the tensor.

    Returns:
        A new Python object with the same structure as `obj` but where the tensors are now Numpy
        arrays. Not supported type are left as reference in the new object.

    Example:
        .. code-block:: python

            >>> from poutyne import torch_to_numpy
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
        :meth:`~poutyne.torch_apply` for supported types.
    """
    if copy:

        def func(t):
            return t.detach().cpu().numpy().copy()

    else:

        def func(t):
            return t.detach().cpu().numpy()

    return torch_apply(obj, func)


def torch_to(obj, *args, **kargs):
    return torch_apply(obj, lambda t: t.to(*args, **kargs))


def torch_apply(obj, func):
    """
    Apply a function to all tensors inside a Python object composed of the supported types.

    Supported types are: list, tuple and dict.

    Args:
        obj: The Python object to convert.
        func: The function to apply.

    Returns:
        A new Python object with the same structure as `obj` but where the tensors have been applied
        the function `func`. Not supported type are left as reference in the new object.
    """

    def fn(t):
        return func(t) if torch.is_tensor(t) else t

    return _apply(obj, fn)


def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        if isinstance(obj, PackedSequence):
            return type(obj)(
                *(_apply(getattr(obj, el), func) if el != "batch_sizes" else getattr(obj, el) for el in obj._fields)
            )
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: _apply(el, func) for k, el in obj.items()}
    return func(obj)


def _concat(obj):
    first_item = obj[0]
    if isinstance(first_item, (list, tuple)):
        return type(first_item)(_concat(ele) for ele in zip(*obj))
    if isinstance(first_item, dict):
        concat_dict = {}
        for key in first_item.keys():
            concat_dict[key] = _concat([o[key] for o in obj])
        return concat_dict
    if isinstance(first_item, np.ndarray) and len(first_item.shape) != 0:
        return np.concatenate(obj)
    if torch.is_tensor(first_item) and len(first_item.shape) != 0:
        return torch.cat(obj)
    return obj


def numpy_to_torch(obj):
    """
    Convert to tensors all Numpy arrays inside a Python object composed of the supported types.

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the Numpy arrays are now
        tensors. Not supported type are left as reference in the new object.

    Example:
        .. code-block:: python

            >>> from poutyne import numpy_to_torch
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

    def fn(a):
        return torch.from_numpy(a) if isinstance(a, np.ndarray) else a

    return _apply(obj, fn)


def _assert_and_get_length_recursively(obj):
    if isinstance(obj, (list, tuple)):
        lengths = [_assert_and_get_length_recursively(o) for o in obj]
        for length in lengths[1:]:
            assert length == lengths[0]
        return lengths[0]
    return len(obj)


def _get_index_recursively(obj, idx):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_get_index_recursively(o, idx) for o in obj)
    return obj[idx]


class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        super().__init__()
        self.tensors = tensors
        self._len = _assert_and_get_length_recursively(self.tensors)

    def __getitem__(self, index):
        return _get_index_recursively(self.tensors, index)

    def __len__(self):
        return self._len


def set_seeds(seed):
    """
    Set Python, Numpy and Pytorch's random seeds in order to make
    the random number generation procedure deterministic and reproducible.

    Args:
        seed (int): The random number generation seed to use.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def save_random_states(f: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    """
    Save Python, Numpy and Pytorch's (both CPU and GPU) random states.

    Args:
        f (Union[str, os.PathLike, BinaryIO, IO[bytes]]): a file-like object (has to implement write and flush) or
            a string or os.PathLike object containing a file name.
    """
    torch.save(
        dict(
            cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
            numpy=np.random.get_state(),
            python=random.getstate(),
        ),
        f,
    )


def load_random_states(f: Any):
    """
    Load Python, Numpy and Pytorch's (both CPU and GPU) random states as saved by :func:`~poutyne.save_random_states()`.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
    """
    states = torch.load(f)
    torch.set_rng_state(states["cpu"])
    torch.cuda.set_rng_state_all(states["cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])


def is_in_jupyter_notebook():
    # pylint: disable=import-outside-toplevel
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        jupyter = shell in ['ZMQInteractiveShell', 'Shell']
    except ImportError:
        jupyter = False

    return jupyter


def get_batch_size(*values):
    """
    This method infers the batch size of a batch. Here is the inferring algorithm used to compute the
    batch size. The values are tested in order at each step of the inferring algorithm. If one
    step succeed for one of the values, the algorithm stops.

    - Step 1: if a value is a tensor or a Numpy array, then the ``len()`` is returned.
    - Step 2: if a value is a list or a tuple, then the ``len()`` of the first element is returned
      if it is a tensor or a Numpy array.
    - Step 3: if a value is a dict, then the value for the key ``'batch_size'`` is returned if it
      is of integral type.
    - Step 4: if a value is a dict, then the ``len()`` of the first element of ``.values()`` is
      returned if it is a tensor or a Numpy array.

    If inferring the batch size is not possible, the batch size is set to 1 and a warning is raised.
    To disable this warning, set

    .. code-block:: python

        from poutyne import warning_settings\n
        warning_settings['batch_size'] = 'ignore'\n\n

    Args:
        values: The values used for inferring the batch size.
    """

    def is_torch_or_numpy(v):
        return torch.is_tensor(v) or isinstance(v, np.ndarray)

    for v in values:
        if is_torch_or_numpy(v):
            return len(v)
    for v in values:
        if isinstance(v, (tuple, list)):
            if is_torch_or_numpy(v[0]):
                return len(v[0])
    for v in values:
        if isinstance(v, dict):
            if 'batch_size' in v and isinstance(v['batch_size'], numbers.Integral):
                return v['batch_size']
    for v in values:
        if isinstance(v, dict):
            first_value = list(v.values())[0]
            if is_torch_or_numpy(first_value):
                return len(first_value)

    if warning_settings['batch_size'] == 'warn':
        warnings.warn(
            "Inferring the batch size is not possible. Hence, the batch size is set to 1. To disable this warning, "
            "set\n"
            "from poutyne import warning_settings\n"
            "warning_settings['batch_size'] = 'ignore'\n\n"
            #
            #
            "Here is the inferring algorithm used to compute the batch size. The values are tested in order at each "
            "step of the inferring algorithm. If one step succeed for one of the values, the algorithm stops.\n\n"
            #
            #
            "Step 1: if a value is a tensor or a Numpy array, then the 'len()' is returned.\n"
            #
            "Step 2: if a value is a list or a tuple, then the 'len()' of the first element is returned if it is a "
            "tensor or a Numpy array.\n"
            #
            "Step 3: if a value is a dict, then the value for the key 'batch_size' is returned if it is of integral "
            "type.\n"
            #
            "Step 4: if a value is a dict, then the 'len()' of the first element of '.values()' is returned if it is a "
            "tensor or a Numpy array.\n"
        )
    return 1
